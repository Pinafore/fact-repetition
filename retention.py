#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import pickle
import codecs
import argparse
import itertools
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from collections import defaultdict
from joblib import Parallel, delayed
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from util import parse_date
from util import User, Card


def apply_parallel(f, groupby):
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(f)(group) for name, group in tqdm(groupby))


def load_protobowl():
    root_dir = '/fs/clip-quiz/shifeng/karl/data/protobowl/'
    log_dir = root_dir + 'protobowl-042818.log'
    raw_df_dir = root_dir + 'protobowl-042818.log.h5'
    filtered_df_dir = root_dir + 'protobowl-042818.log.filtered.h5'
    questions_dir = root_dir + 'protobowl-042818.log.questions.pkl'

    if os.path.exists(filtered_df_dir):
        with pd.HDFStore(filtered_df_dir) as store:
            df = store['data']
        return df

    if not os.path.exists(raw_df_dir):
        # parse protobowl json log
        data = []
        line_count = 0
        questions = dict()  # qid -> text
        pbar = tqdm(total=5137085)
        with codecs.open(log_dir, 'r', 'utf-8') as f:
            line = f.readline()
            pbar.update(1)
            while line is not None:
                line = line.strip()
                if len(line) < 1:
                    break
                while not line.endswith('}}'):
                    _line = f.readline()
                    pbar.update(1)
                    if _line is None:
                        break
                    line += _line.strip()
                try:
                    line = json.loads(line)
                except ValueError:
                    line = f.readline()
                    pbar.update(1)
                    if line is None:
                        break
                    continue
                line_count += 1
                if line_count % 10000 == 0:
                    sys.stderr.write('\rdone: {}/5130000'.format(line_count))

                obj = line['object']
                date = datetime.strptime(line['date'][:-6],
                                         '%a %b %d %Y %H:%M:%S %Z%z')
                relative_position = obj['time_elapsed'] / obj['time_remaining']

                data.append((
                    date, obj['qid'], obj['user']['id'],
                    relative_position, obj['guess'], obj['ruling']))

                if obj['qid'] not in questions:
                    questions[obj['qid']] = obj['question_text']

                line = f.readline()
                pbar.update(1)
        pbar.close()
        df = pd.DataFrame(
            data,
            columns=['date', 'qid', 'uid', 'buzzing_position', 'guess', 'result'])

        with pd.HDFStore(raw_df_dir) as store:
            store['data'] = df
        with open(questions_dir, 'wb') as f:
            pickle.dump(questions, f)
    else:
        with pd.HDFStore(raw_df_dir) as store:
            df = store['data']

    print('remove users who answered fewer than 10 questions')
    df = df.groupby('uid').filter(lambda x: len(x.groupby('qid')) >= 10)

    print('remove duplicate records within n hrs')
    df = filter_by_time(df)

    print(len(set(df.uid)), 'users')
    print(len(set(df.qid)), 'questions')
    print(len(df), 'records')

    with pd.HDFStore(filtered_df_dir) as store:
        store['data'] = df

    return df


def split_train_test(df):
    root_dir = '/fs/clip-quiz/shifeng/karl/data/protobowl/'
    train_df_dir = root_dir + 'protobowl-042818.log.train.h5'
    test_df_dir = root_dir + 'protobowl-042818.log.test.h5'

    if os.path.exists(train_df_dir) and os.path.exists(test_df_dir):
        with pd.HDFStore(train_df_dir) as store:
            train_df = store['data']
        with pd.HDFStore(test_df_dir) as store:
            test_df = store['data']
        return {'train': train_df, 'test': test_df}

    print('split users by date of first appearance')
    train_df, test_df = split_uids_by_date(df)

    # save dataframe
    with pd.HDFStore(train_df_dir) as store:
        store['data'] = train_df
    with pd.HDFStore(test_df_dir) as store:
        store['data'] = test_df

    return train_df, test_df


def group_filter_by_time(group):
    # for uid records, only keep the first qid within time frame
    time_window = 60 * 60 * 1  # 1hr
    group = group.sort_values('date')
    current_date = parse_date('1910-06-1 18:27:08.172341')
    index_to_drop = []
    for row in group.itertuples(index=True):
        date = row.date.replace(tzinfo=None)
        if (date - current_date).seconds <= time_window:
            index_to_drop.append(row.Index)
        else:
            current_date = date
    return index_to_drop


def filter_by_time(df):
    # only keep the first appearance of (uid, qid) within time frame
    dedup_index_dir = 'data/protobowl/dedup_2hr_index.json'
    if os.path.exists(dedup_index_dir):
        print('loading drop index')
        with open(dedup_index_dir) as f:
            index_to_drop = json.load(f)
    else:
        # this will take a while
        print('creating drop index')
        index_to_drop_list = apply_parallel(group_filter_by_time,
                                            df.groupby(['qid', 'uid']))
        index_to_drop = list(itertools.chain(*index_to_drop_list))
        with open(dedup_index_dir, 'w') as f:
            json.dump(index_to_drop, f)
    df = df.drop(index_to_drop, axis='index')
    return df


def get_first_appearance_date(group):
    group = group.sort_values('date')
    return group.iloc[0]['uid'], group.iloc[0]['date']


def split_uids_by_date(df):
    # order and split users by first appearance dates
    df_by_uid = df.groupby('uid')
    returns = apply_parallel(get_first_appearance_date, df_by_uid)
    returns = sorted(returns, key=lambda x: x[1])
    uids, first_appearance_dates = list(zip(*returns))
    train_uids = uids[:int(len(uids) * 0.7)]
    test_uids = uids[int(len(uids) * 0.7):]
    train_index = list(itertools.chain(*[
        df_by_uid.get_group(uid).index.tolist() for uid in train_uids]))
    test_index = list(itertools.chain(*[
        df_by_uid.get_group(uid).index.tolist() for uid in test_uids]))
    train_df = df.loc[train_index]
    test_df = df.loc[test_index]
    return train_df, test_df


def accumulative_user_features(group):
    # for each user, order records by date for accumulative features
    previous_date = {}
    overall_results = []  # keep track of average accuracy
    question_results = defaultdict(list)  # keep track of average accuracy
    count_correct_of_qid = defaultdict(lambda: 0)  # keep track of repetition
    count_wrong_of_qid = defaultdict(lambda: 0)  # keep track of repetition
    count_total_of_qid = defaultdict(lambda: 0)  # keep track of repetition

    # below are returned feature values
    index = []
    count_correct = []
    count_wrong = []
    count_total = []
    average_overall_accuracy = []
    average_question_accuracy = []
    previous_result = []
    gap_from_previous = []

    for row in group.sort_values('date').itertuples():
        index.append(row.Index)
        if len(question_results[row.qid]) == 0:
            # first time answering qid
            count_correct.append(0)
            count_wrong.append(0)
            count_total.append(0)
            average_question_accuracy.append(0)
            previous_result.append(0)
            gap_from_previous.append(0)
        else:
            count_correct.append(count_correct_of_qid[row.qid])
            count_wrong.append(count_wrong_of_qid[row.qid])
            count_total.append(count_total_of_qid[row.qid])
            average_question_accuracy.append(np.mean(question_results[row.qid]))
            previous_result.append(question_results[row.qid][-1])
            gap_from_previous.append((row.date - previous_date[row.qid]).seconds / (60 * 60))

        if len(overall_results) == 0:
            average_overall_accuracy.append(0)
        else:
            average_overall_accuracy.append(np.mean(overall_results))

        # result = True, False, or prompt
        previous_date[row.qid] = row.date
        overall_results.append(row.result_binary)
        question_results[row.qid].append(row.result_binary)
        count_correct_of_qid[row.qid] += row.result_binary
        count_wrong_of_qid[row.qid] += (1 - row.result_binary)
        count_total_of_qid[row.qid] += 1

    return (
        index,
        count_correct,
        count_wrong,
        count_total,
        average_overall_accuracy,
        average_question_accuracy,
        previous_result,
        gap_from_previous
    )


def accumulative_question_features(group):
    overall_results = []
    index = []
    average_overall_accuracy = []
    count_total = []
    count_correct = []
    count_wrong = []
    for row in group.sort_values('date').itertuples():
        index.append(row.Index)
        if len(overall_results) == 0:
            average_overall_accuracy.append(0)
        else:
            average_overall_accuracy.append(np.mean(overall_results))
        count_total.append(len(overall_results))
        count_correct.append(sum(overall_results))
        count_wrong.append(count_total[-1] - count_correct[-1])
        overall_results.append(row.result_binary)
    return (
        index,
        average_overall_accuracy,
        count_total,
        count_correct,
        count_wrong
    )


def featurize(df):
    x_train_dir = 'data/protobowl/x_train.npy'
    y_train_dir = 'data/protobowl/y_train.npy'
    x_test_dir = 'data/protobowl/x_test.npy'
    y_test_dir = 'data/protobowl/y_test.npy'
    dirs = [x_train_dir, y_train_dir, x_test_dir, y_test_dir]

    if all(os.path.exists(d) for d in dirs):
        x_train = np.load(x_train_dir)
        y_train = np.load(y_train_dir)
        x_test = np.load(x_test_dir)
        y_test = np.load(y_test_dir)
        return x_train, y_train, x_test, y_test

    # result = True, False, or prompt
    df['result_binary'] = df['result'].map(lambda x: 1 if x is True else 0)

    '''
    number_of_records_by_uid = df.groupby('uid').size()
    # df['user_count'] = df.uid.map(number_of_records_by_uid.to_dict())
    print('average number of questions answered by each user',
          number_of_records_by_uid.mean())

    number_of_records_by_qid = df.groupby('qid').size()
    # df['question_count'] = df.qid.map(number_of_records_by_qid.to_dict())
    print('average number of users that answered each question',
          number_of_records_by_qid.mean())

    number_of_records_by_uid_qid = df.groupby(['uid', 'qid']).size()
    print('average repetition of qid + uid',
          number_of_records_by_uid_qid.mean())

    accuracy_by_uid = df[['uid', 'result_binary']].groupby('uid').agg('mean').result_binary
    # df['user_accuracy'] = df.uid.map(accuracy_by_uid.to_dict())
    print('average user accuracy', accuracy_by_uid.mean())
    accuracy_by_qid = df[['qid', 'result_binary']].groupby('qid').agg('mean').result_binary
    # df['question_accuracy'] = df.qid.map(accuracy_by_qid.to_dict())
    print('average question accuracy', accuracy_by_qid.mean())
    '''

    user_features = apply_parallel(accumulative_user_features, df.groupby('uid'))
    user_features = list(zip(*user_features))
    user_features = [itertools.chain(*x) for x in user_features]
    # convert generator to list here since it's used multiple times
    user_index = list(user_features[0])
    user_features = user_features[1:]  # skip index
    user_features = [{i: v for i, v in zip(user_index, f)} for f in user_features]
    user_feature_names = [
        'user_count_correct',
        'user_count_wrong',
        'user_count_total',
        'user_average_overall_accuracy',
        'user_average_question_accuracy',
        'user_previous_result',
        'user_gap_from_previous',
    ]
    for name, feature in zip(user_feature_names, user_features):
        df[name] = df.index.map(feature)

    question_features = apply_parallel(accumulative_question_features, df.groupby('qid'))
    question_features = list(zip(*question_features))
    question_features = [itertools.chain(*x) for x in question_features]
    # convert generator to list here since it's used multiple times
    question_index = list(question_features[0])
    question_features = question_features[1:]  # skip index
    question_features = [{i: v for i, v in zip(question_index, f)} for f in question_features]
    question_feature_names = [
        'question_average_overall_accuracy',
        'question_count_total',
        'question_count_correct',
        'question_count_wrong',
    ]
    for name, feature in zip(question_feature_names, question_features):
        df[name] = df.index.map(feature)

    df['bias'] = 1

    train_df, test_df = split_train_test(df)

    feature_names = user_feature_names + question_feature_names + ['bias']
    x_train = train_df[feature_names].to_numpy().astype(np.float32)
    y_train = train_df['result_binary'].to_numpy().astype(int)
    x_test = test_df[feature_names].to_numpy().astype(np.float32)
    y_test = test_df['result_binary'].to_numpy().astype(int)

    np.save(x_train_dir, x_train)
    np.save(y_train_dir, y_train)
    np.save(x_test_dir, x_test)
    np.save(y_test_dir, y_test)

    return x_train, y_train, x_test, y_test


class RetentionDataset(torch.utils.data.Dataset):

    def __init__(self, fold='train'):
        x_train, y_train, x_test, y_test = featurize(load_protobowl())

        data = {
            'train': (x_train, y_train),
            'test': (x_test, y_test)
        }

        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)
        self.mean[-1] = 0
        self.std[-1] = 1

        self.x, self.y = data[fold]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = (self.x[idx] - self.mean) / self.std
        y = np.array(self.y[idx])
        return torch.from_numpy(x), torch.from_numpy(y)


class Net(nn.Module):
    def __init__(self, n_input):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, 128)
        self.dropout1 = nn.Dropout(0.25)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    loss_func = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_func(logits, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    loss_func = nn.CrossEntropyLoss(reduction='sum')
    predictions = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)
            # sum up batch loss
            test_loss += loss_func(logits, target).item()
            # get the index of the max log-probability
            pred = logits.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            predictions += pred[:, 0].detach().cpu().numpy().tolist()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, predictions


def main():
    parser = argparse.ArgumentParser(description='Retention model')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_dataset = RetentionDataset('train')
    test_dataset = RetentionDataset('test')
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    n_input = train_dataset.x.shape[1]
    model = Net(n_input=n_input).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_test_loss = 9999
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss, predictions = test(args, model, device, test_loader)
        scheduler.step()
        if test_loss < best_test_loss:
            if args.save_model:
                checkpoint_dir = "checkpoints/retention_model.pt"
                torch.save(model.state_dict(), checkpoint_dir)
                print('save model checkpoint to', checkpoint_dir)
            best_test_loss = test_loss


class RetentionModel:

    def __init__(self, use_cuda=True):
        self.dataset = RetentionDataset()
        n_input = self.dataset.x.shape[1]
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = Net(n_input=n_input).to(self.device)
        self.model.load_state_dict(torch.load('checkpoints/retention_model.pt'))
        self.model.eval()

    def predict(self, user: User, card: Card, date=None):
        # TODO batch version
        if date is None:
            date = datetime.now()
        # user_count_correct
        # user_count_wrong
        # user_count_total
        # user_average_overall_accuracy
        # user_average_question_accuracy
        # user_previous_result
        # user_gap_from_previous
        # question_average_overall_accuracy
        # question_count_total
        # question_count_correct
        # question_count_wrong
        # bias
        uq_correct = user.count_correct_before.get(card.card_id, 0)
        uq_wrong = user.count_wrong_before.get(card.card_id, 0)
        uq_total = uq_correct + uq_wrong
        x = np.array([
            uq_correct,  # user_count_correct
            uq_wrong,  # user_count_wrong
            uq_total,  # user_count_total
            0 if len(user.results) == 0 else np.mean(user.results),  # user_average_overall_accuracy
            0 if uq_total == 0 else uq_correct / uq_total,  # user_average_question_accuracy
            0 if len(user.results) == 0 else user.results[-1],  # user_previous_result
            (date - user.last_study_date.get(card.card_id, date)).seconds / (60 * 60),  # user_gap_from_previous
            0 if len(card.results) == 0 else np.mean(card.results),  # question_average_overall_accuracy
            len(card.results),  # question_count_total
            sum(card.results),  # question_count_correct
            len(card.results) - sum(card.results),  # question_count_wrong
            1  # bias
        ]).astype(np.float32)
        x = (x - self.dataset.mean) / self.dataset.std
        x = x[np.newaxis, :]
        x = torch.from_numpy(x).to(self.device)
        logits = self.model.forward(x)
        return F.softmax(logits, dim=1).detach().cpu().numpy()[0]


def unit_test():
    user = User(
        user_id='user 1',
        qrep=[np.array([0.1, 0.2, 0.3])],
        skill=[np.array([0.1, 0.2, 0.3])],
        category='History',
        last_study_date={'card 1': datetime.now()},
        leitner_box={'card 1': 2},
        leitner_scheduled_date={'card 2': datetime.now()},
        sm2_efactor={'card 1': 0.5},
        sm2_interval={'card 1': 6},
        sm2_repetition={'card 1': 10},
        sm2_scheduled_date={'card 2': datetime.now()},
        results=[True, False, True],
        count_correct_before={'card 1': 1},
        count_wrong_before={'card 1': 3}
    )

    card = Card(
        card_id='card 1',
        text='This is the question text',
        answer='Answer Text III',
        category='WORLD',
        qrep=np.array([1, 2, 3, 4]),
        skill=np.array([0.1, 0.2, 0.3, 0.4]),
        results=[True, False, True, True]
    )

    model = RetentionModel()
    print(model.predict(user, card))


def test_model():
    parser = argparse.ArgumentParser(description='Retention model')
    parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=6, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                        help='how many batches to wait before logging status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_dataset = RetentionDataset('train')
    test_dataset = RetentionDataset('test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    n_input = train_dataset.x.shape[1]
    model = Net(n_input=n_input).to(device)
    model.load_state_dict(torch.load('checkpoints/retention_model.pt'))
    model.eval()

    test_loss, predictions = test(args, model, device, test_loader)
    print(predictions)


if __name__ == '__main__':
    # main()
    # unit_test()
    test_model()
