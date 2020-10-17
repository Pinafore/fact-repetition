#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
from typing import List
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from karl.new_util import User, Fact, parse_date
from karl.retention.data import RetentionDataset, apply_parallel, get_split_dfs


class Net(nn.Module):

    def __init__(self, n_input):
        super(Net, self).__init__()
        # self.fc1 = nn.Linear(n_input, 128)
        # self.dropout1 = nn.Dropout(0.25)
        # self.fc2 = nn.Linear(128, 2)
        self.fc1 = nn.Linear(n_input, 2)

    def forward(self, x):
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout1(x)
        # x = self.fc2(x)
        # return x
        return self.fc1(x)


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
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='skip training')
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

    checkpoint_dir = "checkpoints/retention_model.pt"

    if args.evaluate:
        checkpoint_dir = "checkpoints/retention_model.pt"
        model.load_state_dict(torch.load('checkpoints/retention_model.pt'))
        model.eval()
        test_loss, predictions = test(args, model, device, test_loader)
        return

    best_test_loss = 9999
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test_loss, predictions = test(args, model, device, test_loader)
        scheduler.step()
        if test_loss < best_test_loss:
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
    
    def compute_features(
        self,
        user: User,
        fact: Fact,
        date: datetime,
    ):

        uq_correct = user.count_correct_before.get(fact.fact_id, 0)
        uq_wrong = user.count_wrong_before.get(fact.fact_id, 0)
        uq_total = uq_correct + uq_wrong
        if fact.fact_id in user.previous_study:
            prev_date, prev_response = user.previous_study[fact.fact_id]
        else:
            # TODO this really shouldn't be the current date.
            # the default prev_date should be something much earlier
            prev_date = date
        if isinstance(prev_date, str):
            prev_date = parse_date(prev_date)
        if fact.results is None:
            fact.results = []
        features = [
            uq_correct,  # user_count_correct
            uq_wrong,  # user_count_wrong
            uq_total,  # user_count_total
            0 if len(user.results) == 0 else np.mean(user.results),  # user_average_overall_accuracy
            0 if uq_total == 0 else uq_correct / uq_total,  # user_average_question_accuracy
            0 if len(user.results) == 0 else user.results[-1],  # user_previous_result
            (date - prev_date).seconds / (60 * 60),  # user_gap_from_previous
            0 if len(fact.results) == 0 else np.mean(fact.results),  # question_average_overall_accuracy
            len(fact.results),  # question_count_total
            sum(fact.results),  # question_count_correct
            len(fact.results) - sum(fact.results),  # question_count_wrong
            1  # bias
        ]
        feature_names = [
            'user_count_correct',
            'user_count_wrong',
            'user_count_total',
            'user_average_overall_accuracy',
            'user_average_question_accuracy',
            'user_previous_result',
            'user_gap_from_previous',
            'question_average_overall_accuracy',
            'question_count_total',
            'question_count_correct',
            'question_count_wrong',
            'bias',
        ]
        feature_dict = {k: v for k, v in zip(feature_names, features)}
        return features, feature_dict

    def predict(
        self,
        user: User,
        facts: List[Fact],
        date: datetime = None,
    ) -> np.ndarray:
        if date is None:
            date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
        xs = [self.compute_features(user, fact, date)[0] for fact in facts]
        xs = np.array(xs).astype(np.float32)
        xs = (xs - self.dataset.mean) / self.dataset.std
        ys = []
        batch_size = 128
        for i in range(0, xs.shape[0], batch_size):
            x = xs[i: i + batch_size]
            x = torch.from_numpy(x).to(self.device)
            logits = self.model.forward(x)
            # return the probability of positive (1)
            y = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1]
            ys.append(y)
        return np.concatenate(ys)

    def predict_one(
        self,
        user: User,
        fact: Fact,
        date: datetime
    ) -> float:
        '''recall probability of a single fact'''
        return self.predict(user, [fact], date)[0]  # batch size is 1


def test_wrapper():
    user = User(
        user_id='user 1',
        qrep=[np.array([0.1, 0.2, 0.3])],
        skill=[np.array([0.1, 0.2, 0.3])],
        category='History',
        previous_study={'fact 1': (datetime.now(), True)},
        leitner_box={'fact 1': 2},
        leitner_scheduled_date={'fact 2': datetime.now()},
        sm2_efactor={'fact 1': 0.5},
        sm2_interval={'fact 1': 6},
        sm2_repetition={'fact 1': 10},
        sm2_scheduled_date={'fact 2': datetime.now()},
        results=[True, False, True],
        count_correct_before={'fact 1': 1},
        count_wrong_before={'fact 1': 3}
    )

    fact = Fact(
        fact_id='fact 1',
        text='This is the question text',
        answer='Answer Text III',
        category='WORLD',
        qrep=np.array([1, 2, 3, 4]),
        skill=np.array([0.1, 0.2, 0.3, 0.4]),
        results=[True, False, True, True]
    )

    model = RetentionModel()
    print(1 - model.predict(user, [fact, fact, fact]))
    print(model.predict_one(user, fact))


def test_majority():
    train_df, test_df = get_split_dfs()

    def get_user_majority(group):
        return group.iloc[0]['uid'], int(group['result_binary'].mean() > 0.5)

    returns = apply_parallel(get_user_majority, train_df.groupby('uid'))
    uids, labels = list(zip(*returns))
    test_df['user_majority'] = test_df['uid'].map({x: y for x, y in zip(uids, labels)})
    print('user majority acc train -> test',
          (test_df['user_majority'] == test_df['result_binary']).mean())

    returns = apply_parallel(get_user_majority, test_df.groupby('uid'))
    uids, labels = list(zip(*returns))
    test_df['user_majority'] = test_df['uid'].map({x: y for x, y in zip(uids, labels)})
    print('user majority acc test -> test',
          (test_df['user_majority'] == test_df['result_binary']).mean())

    def get_question_majority(group):
        return group.iloc[0]['qid'], int(group['result_binary'].mean() > 0.5)

    returns = apply_parallel(get_question_majority, train_df.groupby('qid'))
    qids, labels = list(zip(*returns))
    test_df['question_majority'] = test_df['qid'].map({x: y for x, y in zip(qids, labels)})
    print('question majority acc train -> test',
          (test_df['question_majority'] == test_df['result_binary']).mean())

    returns = apply_parallel(get_question_majority, test_df.groupby('qid'))
    qids, labels = list(zip(*returns))
    test_df['question_majority'] = test_df['qid'].map({x: y for x, y in zip(qids, labels)})
    print('question majority acc test -> test',
          (test_df['question_majority'] == test_df['result_binary']).mean())


if __name__ == '__main__':
    main()
    # test_wrapper()
    # test_majority()
