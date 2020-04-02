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
    log_dir = '/fs/clip-quiz/shifeng/karl/data/protobowl/protobowl-042818.log'
    raw_df_dir = '/fs/clip-quiz/shifeng/karl/data/protobowl/protobowl-042818.log.h5'
    train_df_dir = '/fs/clip-quiz/shifeng/karl/data/protobowl/protobowl-042818.log.train.h5'
    test_df_dir = '/fs/clip-quiz/shifeng/karl/data/protobowl/protobowl-042818.log.test.h5'
    questions_dir = '/fs/clip-quiz/shifeng/karl/data/protobowl/protobowl-042818.log.questions.pkl'

    if os.path.exists(train_df_dir) and os.path.exists(test_df_dir):
        with pd.HDFStore(train_df_dir) as f:
            train_df = f['data']
        with pd.HDFStore(test_df_dir) as f:
            test_df = f['data']
        return {'train': train_df, 'test': test_df}

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
                date = datetime.strptime(line['date'][:-6], '%a %b %d %Y %H:%M:%S %Z%z')
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
            data, columns=['date', 'qid', 'uid', 'buzzing_position', 'guess', 'result'])

        with pd.HDFStore(raw_df_dir) as f:
            f['data'] = df
        with open(questions_dir, 'wb') as f:
            pickle.dump(questions, f)
    else:
        with pd.HDFStore(raw_df_dir) as f:
            df = f['data']

    print('remove users who answered fewer than 20 questions')
    df = df.groupby('uid').filter(lambda x: len(x.groupby('qid')) >= 20)

    print('remove duplicate records within n hrs')
    df = filter_by_time(df)

    print('split users by date of first appearance')
    train_df, test_df = split_uids_by_date(df)

    # save dataframe
    with pd.HDFStore(train_df_dir) as f:
        f['data'] = train_df
    with pd.HDFStore(test_df_dir) as f:
        f['data'] = test_df

    print(len(set(df.uid)), 'users')
    print(len(set(df.qid)), 'questions')
    print(len(df), 'records')

    return {'train': train_df, 'test': test_df}


def group_filter_by_time_and_split(group):
    # deduplicate uid + qid group by time
    # for all records with the same uid and qid, only keep the first record
    # every two hours, returns a list of indices to be dropped
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
    dedup_index_dir = 'data/protobowl/dedup_2hr_index.json'
    if os.path.exists(dedup_index_dir):
        print('loading drop index')
        with open(dedup_index_dir) as f:
            index_to_drop = json.load(f)
    else:
        # this will take a while
        print('creating drop index')
        records_by_qid_uid = df.groupby(['qid', 'uid'])
        index_to_drop_list = apply_parallel(group_filter_by_time_and_split,
                                            records_by_qid_uid)
        index_to_drop = list(itertools.chain(*index_to_drop_list))
        with open(dedup_index_dir, 'w') as f:
            json.dump(index_to_drop, f)
    df = df.drop(index_to_drop, axis='index')
    return df


def split_uids_by_date(df):
    # order users by their first appearance
    records_by_uid = df.groupby('uid')
    uids = list(records_by_uid.groups.keys())
    train_uids = uids[:int(0.7 * len(uids))]
    test_uids = uids[int(0.7 * len(uids)):]
    train_index = list(itertools.chain(*[records_by_uid.get_group(uid).index.tolist() for uid in train_uids]))
    test_index = list(itertools.chain(*[records_by_uid.get_group(uid).index.tolist() for uid in test_uids]))
    train_df = df.loc[train_index]
    test_df = df.loc[test_index]
    return train_df, test_df


def count_features(group_of_uid):
    # for each user, order records by date, then create
    # times_seen_correct, times_seen_wrong
    index = []
    count_correct, count_wrong = [], []  # list of feature value
    count_correct_of_qid, count_wrong_of_qid = {}, {}
    for row in group_of_uid.sort_values('date').itertuples():
        if row.qid not in count_correct_of_qid:
            count_correct_of_qid[row.qid] = 0
        if row.qid not in count_wrong_of_qid:
            count_wrong_of_qid[row.qid] = 0
        index.append(row.Index)
        count_correct.append(count_correct_of_qid[row.qid])
        count_wrong.append(count_wrong_of_qid[row.qid])
        if row.result:
            count_correct_of_qid[row.qid] += 1
        else:
            count_wrong_of_qid[row.qid] += 1
    return index, count_correct, count_wrong


def featurize(df):
    df['result_binary'] = df['result'].apply(lambda x: 1 if x else 0)

    number_of_records_by_uid = df.groupby('uid').size()
    df['user_count'] = df.uid.map(number_of_records_by_uid.to_dict())
    print('average number of questions answered by each user', number_of_records_by_uid.mean())

    number_of_records_by_qid = df.groupby('qid').size()
    df['question_count'] = df.qid.map(number_of_records_by_qid.to_dict())
    print('average number of users that answered each question', number_of_records_by_qid.mean())

    number_of_records_by_uid_qid = df.groupby(['uid', 'qid']).size()
    print('average repetition of qid + uid', number_of_records_by_uid_qid.mean())

    accuracy_by_uid = df[['uid', 'result_binary']].groupby('uid').agg('mean').result_binary
    df['user_accuracy'] = df.uid.map(accuracy_by_uid.to_dict())
    print('average user accuracy', accuracy_by_uid.mean())
    accuracy_by_qid = df[['qid', 'result_binary']].groupby('qid').agg('mean').result_binary
    df['question_accuracy'] = df.qid.map(accuracy_by_qid.to_dict())
    print('average question accuracy', accuracy_by_qid.mean())

    count_returns = apply_parallel(count_features, df.groupby('qid'))
    index_list, count_correct_list, count_wrong_list = list(zip(*count_returns))
    index = list(itertools.chain(*index_list))  # used twice
    count_correct = itertools.chain(*count_correct_list)
    count_wrong = itertools.chain(*count_wrong_list)
    index_to_count_correct = {i: c for i, c in zip(index, count_correct)}
    index_to_count_wrong = {i: c for i, c in zip(index, count_wrong)}
    df['count_correct_before'] = df.index.map(index_to_count_correct)
    df['count_wrong_before'] = df.index.map(index_to_count_wrong)

    df['bias'] = 1

    x = df[['user_accuracy', 'question_accuracy',
            'count_correct_before', 'count_wrong_before',
            'bias']].to_numpy()
    y = df['result_binary'].to_numpy()
    return x, y


class RetentionDataset(torch.utils.data.Dataset):

    def __init__(self, fold='train'):
        dirs = {
            'train': ('data/protobowl/x_train.npy', 'data/protobowl/y_train.npy'),
            'test': ('data/protobowl/x_test.npy', 'data/protobowl/y_test.npy')
        }
        x_dir, y_dir = dirs[fold]
        if os.path.exists(x_dir) and os.path.exists(y_dir):
            x, y = np.load(x_dir), np.load(y_dir)
        else:
            x, y = featurize(load_protobowl()[fold])
            np.save(x_dir, x)
            np.save(y_dir, y)
        self.x = x.astype(np.float32)
        self.y = y.astype(int)
        # TODO get statistics from data instead of hard code
        self.mean = np.array([0.62840253, 0.6284026, 0.07828305, 0.04504214, 0.]).astype(np.float32)
        self.std = np.array([0.16344075, 0.21432154, 0.15107092, 0.08828004, 1.]).astype(np.float32)

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
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
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
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

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
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        RetentionDataset('train'),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        RetentionDataset('test'),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)

    model = Net(n_input=5).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_test_loss = 9999
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss = test(args, model, device, test_loader)
        scheduler.step()
        if test_loss < best_test_loss:
            if args.save_model:
                checkpoint_dir = "checkpoints/retention_model.pt"
                torch.save(model.state_dict(), checkpoint_dir)
                print('save model checkpoint to', checkpoint_dir)
            best_test_loss = test_loss


class RetentionModel:

    def __init__(self, use_cuda=True):
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = Net(n_input=5).to(self.device)
        self.model.load_state_dict(torch.load('checkpoints/retention_model.pt'))
        self.model.eval()
        self.mean = np.array([0.62840253, 0.6284026, 0.07828305, 0.04504214, 0.]).astype(np.float32)
        self.std = np.array([0.16344075, 0.21432154, 0.15107092, 0.08828004, 1.]).astype(np.float32)

    def predict(self, user: User, card: Card):
        # 'user_accuracy', 'question_accuracy',
        # 'count_correct_before', 'count_wrong_before'
        x = np.array([
            np.mean(user.results),
            np.mean(card.results),
            user.count_correct_before.get(card.card_id, 0),
            user.count_wrong_before.get(card.card_id, 0),
            1,  # bias
        ]).astype(np.float32)
        x = (x - self.mean) / self.std
        x = (x[np.newaxis, :]).astype(np.float32)
        x = torch.from_numpy(x).to(self.device)
        return self.model.forward(x).argmax().item()

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


if __name__ == '__main__':
    # main()
    # unit_test()
    df = load_protobowl()
