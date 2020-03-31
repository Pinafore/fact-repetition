#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import pickle
import codecs
import argparse
import itertools
import functools
import numpy as np
import pandas as pd
import multiprocessing
from collections import defaultdict
from joblib import Parallel, delayed
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from util import parse_date


def apply_parallel(f, groupby):
    return Parallel(n_jobs=multiprocessing.cpu_count())(delayed(f)(group) for name, group in groupby)


def process_log_line(x):
    '''Process a single line of the log'''
    obj = x['object']
    date = datetime.strptime(x['date'][:-6], '%a %b %d %Y %H:%M:%S %Z%z')
    relative_position = obj['time_elapsed'] / obj['time_remaining']
    return [date,
            obj['guess'],
            obj['qid'],
            obj['time_elapsed'],
            obj['time_remaining'],
            relative_position,
            obj['ruling'],
            obj['user']['id']],\
        obj['qid'],\
        obj['question_text']


# remove duplicate records
def remove_duplicate(df_grouped, uid):
    '''For each user, only take the first record for each question'''
    group = df_grouped.get_group(uid)
    user_questions = set()
    index = group.date.sort_values()
    rows = []
    for _, row in group.loc[index.index].iterrows():
        if row.qid in user_questions:
            continue
        user_questions.add(row.qid)
        rows.append(row)
    for j, row in enumerate(rows):
        rows[j].user_n_records = len(rows)
    return rows


def load_protobowl(
        protobowl_dir='data/protobowl/protobowl-042818.log',
        min_user_questions=20,
        get_questions=False):
    '''Parse protobowl log, return buzz data and questions.
    Filter users that answered less than `min_user_questions` questions.
    Remove duplicates: for each user, only keep the first record for each
    question.

    Args
        protobowl_dir: json log
        min_user_questions: minimum number of questions answered
    Return
        df: dataframe of buzzing records
        questions: protobowl questions
    '''
    df_dir = protobowl_dir + '.h5'
    question_dir = protobowl_dir + '.questions.pkl'

    if os.path.exists(df_dir) and os.path.exists(df_dir):
        with pd.HDFStore(df_dir) as store:
            df = store['data']
        with open(question_dir, 'rb') as f:
            questions = pickle.load(f)
        if get_questions:
            return df, questions
        else:
            return df

    # parse protobowl json log
    data = []
    count = 0
    user_questions = defaultdict(set)
    questions = dict()
    with codecs.open(protobowl_dir, 'r', 'utf-8') as f:
        line = f.readline()
        while line is not None:
            line = line.strip()
            if len(line) < 1:
                break
            while not line.endswith('}}'):
                _line = f.readline()
                if _line is None:
                    break
                line += _line.strip()
            try:
                line = json.loads(line)
            except ValueError:
                line = f.readline()
                if line is None:
                    break
                continue
            count += 1
            if count % 10000 == 0:
                sys.stderr.write('\rdone: {}/5130000'.format(count))
            x, qid, question_text = process_log_line(line)
            if qid not in questions:
                questions[qid] = question_text
            user_questions[x[-1]].add(qid)  # x[-1] is uid
            data.append(x)
            line = f.readline()

    # filter users without enough questions
    filtered_data = []
    for x in data:
        uid = x[-1]
        if len(user_questions[uid]) >= min_user_questions:
            x.append(len(user_questions[uid]))
            filtered_data.append(x)

    df = pd.DataFrame(
        filtered_data,
        columns=['date', 'guess', 'qid', 'time_elapsed', 'time_remaining',
                 'relative_position', 'result', 'uid', 'user_n_records'])

    df_grouped = df.groupby('uid')
    uids = list(df_grouped.groups.keys())
    pool = multiprocessing.Pool(8)
    _remove_duplicate = functools.partial(remove_duplicate, df_grouped)
    user_rows = pool.map(_remove_duplicate, uids)
    df = pd.DataFrame(list(itertools.chain(*user_rows)), columns=df.columns)
    df_grouped = df.groupby('uid')

    print('{} users'.format(len(df_grouped)))
    print('{} records'.format(len(df)))
    print('{} questions'.format(len(set(df.qid))))

    # save
    with pd.HDFStore(df_dir) as store:
        store['data'] = df
    with open(question_dir, 'wb') as f:
        pickle.dump(questions, f)
    if get_questions:
        return df, questions
    else:
        return df


def filter_group_by_time(group):
    # deduplicate uid + qid group by time
    # for all records with the same uid and qid, only keep the first record
    # every two hours, returns a list of indices to be dropped
    time_window = 60 * 60 * 2  # 2hr
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

def filter_df_by_time(df):
    '''
    Data construction
    - group records by user, for each user:
        - remove duplicate questions within 2hr
        - split by time
        - sort by time and generate features
    - group records by question, for each question:
        - group by user, for each user
            - only keep the first record every 2hr
        - split users randomly, for each user
            - sort records by time, generate features
    '''
    dedup_index_dir = 'data/protobowl/dedup_2hr_index.json'
    if os.path.exists(dedup_index_dir):
        print('loading drop index')
        with open(dedup_index_dir) as f:
            index_to_drop = json.load(f)
    else:
        # this will take a while
        print('creating drop index')
        records_by_qid_uid = df.groupby(['qid', 'uid'])
        index_to_drop_list = apply_parallel(filter_group_by_time, records_by_qid_uid)
        index_to_drop = list(itertools.chain(*index_to_drop_list))
        with open(dedup_index_dir, 'w') as f:
            json.dump(index_to_drop, f)
    df = df.drop(index_to_drop, axis='index')
    records_by_uid = df.groupby('uid')
    uids = list(records_by_uid.groups.keys())
    train_uids = uids[:int(0.7 * len(uids))]
    test_uids = uids[int(0.7 * len(uids)):]
    train_index = list(itertools.chain(*[records_by_uid.get_group(uid).index.tolist() for uid in train_uids]))
    test_index = list(itertools.chain(*[records_by_uid.get_group(uid).index.tolist() for uid in test_uids]))
    train_df = df.loc[train_index]
    test_df = df.loc[test_index]
    return {'train': train_df, 'test': test_df}

def count_features(group_of_uid):
    # for each user, order records by date, then create
    # times_seen_correct, times_seen_wrong
    index = []
    count_correct, count_wrong = [], []  # list of feature value
    count_correct_of_qid = defaultdict(lambda: 0)
    count_wrong_of_qid = defaultdict(lambda: 0)
    for row in group_of_uid.sort_values('date').itertuples():
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
    df['correct_count_before'] = df.index.map(index_to_count_correct)
    df['wrong_count_before'] = df.index.map(index_to_count_wrong)

    x = df[['user_count', 'question_count',
            'user_accuracy', 'question_accuracy',
            'correct_count_before', 'wrong_count_before'
            ]].to_numpy()
    y = df['result_binary'].to_numpy()
    return x, y

class RetentionDataset(torch.utils.data.Dataset):

    def __init__(self, fold='train'):
        """
        Args:
            root_dir (string): Directory with all the images.
        """
        dirs = {
            'train': ('data/protobowl/x_train.npy', 'data/protobowl/y_train.npy'),
            'test': ('data/protobowl/x_test.npy', 'data/protobowl/y_test.npy')
        }
        x_dir, y_dir = dirs[fold]
        if os.path.exists(x_dir) and os.path.exists(y_dir):
            self.x = np.load(x_dir).astype(np.float32)
            self.y = np.load(y_dir).astype(int)
        else:
            df = load_protobowl()
            self.x, self.y = featurize(filter_df_by_time(df)[fold])
            np.save(x_dir, self.x)
            np.save(y_dir, self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return torch.from_numpy(self.x[idx]), torch.from_numpy(np.array(self.y[idx]))


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6, 128)
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

    model = Net().to(device)
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


if __name__ == '__main__':
    main()
