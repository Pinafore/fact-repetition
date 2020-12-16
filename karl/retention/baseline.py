#!/usr/bin/env python
# coding: utf-8

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

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


class TemperatureScaledNet(nn.Module):

    def __init__(self, n_input):
        super(TemperatureScaledNet, self).__init__()
        self.net = Net(n_input)
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x):
        logits = self.net(x)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature


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


def set_temperature(args, model, device, test_loader):
    """
    Tune the tempearature of the model (using the validation set).
    We're going to set it to optimize NLL.
    valid_loader (DataLoader): validation set loader
    """
    model.train()
    loss_func = nn.CrossEntropyLoss(reduction='sum')

    # First: collect all the logits and labels for the validation set
    # do not pass through temperature_scale here
    logits_list = []
    labels_list = []
    with torch.no_grad():
        for input, label in test_loader:
            input, label = input.to(device), label.to(device)
            logits = model.net(input)
            logits_list.append(logits)
            labels_list.append(label)
        logits = torch.cat(logits_list).cuda()
        labels = torch.cat(labels_list).cuda()

    before_temperature_nll = loss_func(model.temperature_scale(logits), labels).item()

    # Next: optimize the temperature w.r.t. NLL
    def eval():
        loss = loss_func(model.temperature_scale(logits), labels)
        loss.backward()
        return loss

    optimizer = optim.LBFGS([model.temperature], lr=0.01, max_iter=50)
    optimizer.step(eval)

    after_temperature_nll = loss_func(model.temperature_scale(logits), labels).item()

    print('Optimal temperature: %.3f' % model.temperature.item())
    print('Before NLL: %.3f' % before_temperature_nll)
    print('After NLL: %.3f' % after_temperature_nll)


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
    model = TemperatureScaledNet(n_input=n_input).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    checkpoint_dir = "checkpoints/retention_model_20201028.pt"

    if args.evaluate:
        model.load_state_dict(torch.load(checkpoint_dir))
        model.eval()
        test_loss, predictions = test(args, model, device, test_loader)
        set_temperature(args, model, device, test_loader)
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

    def __init__(
        self,
        use_cuda=True,
        checkpoint_dir='checkpoints/retention_model.pt',
    ):
        self.dataset = RetentionDataset()
        n_input = self.dataset.x.shape[1]
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = TemperatureScaledNet(n_input=n_input).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_dir))
        self.model.eval()


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
