import os
import sys
import json
import pickle
import codecs
import random
import logging
import itertools
import numpy as np
import pandas as pd
import multiprocessing
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from joblib import Parallel, delayed
from typing import Iterator, Dict, Optional

from allennlp.data import Instance
from allennlp.data.fields import Field, TextField, ArrayField, LabelField
from allennlp.data.tokenizers import Tokenizer, Token
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer


log = logging.getLogger(__name__)

DATA_DIR = '/fs/clip-quiz/shifeng/karl/data/protobowl/'


def apply_parallel(f, groupby):
    log.info('    apply parallel {}'.format(f.__name__))
    return Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(f)(group) for name, group in tqdm(groupby))


def parse_date(date: str):
    if isinstance(date, datetime):
        return date
    if isinstance(date, str):
        return datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    else:
        raise TypeError("unrecognized type for parse_date")


def get_questions():
    questions_dir = os.path.join(DATA_DIR, 'protobowl-042818.log.questions.pkl')

    if os.path.exists(questions_dir):
        with open(questions_dir, 'rb') as f:
            return pickle.load(f)

    get_raw_df()
    with open(questions_dir, 'rb') as f:
        return pickle.load(f)


def get_raw_df():
    log_dir = os.path.join(DATA_DIR, 'protobowl-042818.log')
    raw_df_dir = os.path.join(DATA_DIR, 'protobowl-042818.log.h5')
    questions_dir = os.path.join(DATA_DIR, 'protobowl-042818.log.questions.pkl')

    if os.path.exists(raw_df_dir):
        log.info('loading raw_df')
        return pd.read_hdf(raw_df_dir)

    log.info('generating raw_df')
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

    df.to_hdf(raw_df_dir, 'data')
    with open(questions_dir, 'wb') as f:
        pickle.dump(questions, f)

    return df


def get_filtered_df():
    filtered_df_dir = os.path.join(DATA_DIR, 'protobowl-042818.log.filtered.h5')

    if os.path.exists(filtered_df_dir):
        log.info('loading filtered_df')
        return pd.read_hdf(filtered_df_dir)

    df = get_raw_df()

    log.info('generating filtered_df')
    log.info('    remove users who answered fewer than 10 questions')
    df = df.groupby('uid').filter(lambda x: len(x.groupby('qid')) >= 10)

    def _filter_by_time(group):
        # for uid records, only keep the first qid within time frame
        time_window = 5 * 60  # 5min
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

    log.info('    remove duplicate appearances of (uid, qid) within time frame')
    index_to_drop_list = apply_parallel(_filter_by_time, df.groupby(['qid', 'uid']))
    index_to_drop = list(itertools.chain(*index_to_drop_list))
    df = df.drop(index_to_drop, axis='index')

    df.to_hdf(filtered_df_dir, 'data')

    return df


def _user_features(group):
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


def _question_features(group):
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


def get_featurized_df():
    featurized_df_dir = os.path.join(DATA_DIR, 'protobowl-042818.log.features.h5')

    if os.path.exists(featurized_df_dir):
        log.info('loading featurized_df')
        return pd.read_hdf(featurized_df_dir)

    df = get_filtered_df()

    log.info('generating featurized_df')
    # result = True, False, or prompt
    df['result_binary'] = df['result'].map(lambda x: 1 if x is True else 0)

    log.info('    creating user accumulative features')
    user_features = apply_parallel(_user_features, df.groupby('uid'))
    user_features = list(zip(*user_features))
    user_features = [itertools.chain(*x) for x in user_features]
    # convert generator to list here since it's used multiple times
    user_index = list(user_features[0])
    user_features = user_features[1:]  # skip index
    user_features = [{i: v for i, v in zip(user_index, f)} for f in user_features]
    user_feature_names = [
        'f_user_count_correct',
        'f_user_count_wrong',
        'f_user_count_total',
        'f_user_average_overall_accuracy',
        'f_user_average_question_accuracy',
        'f_user_previous_result',
        'f_user_gap_from_previous',
    ]
    for name, feature in zip(user_feature_names, user_features):
        df[name] = df.index.map(feature)

    log.info('    creating question accumulative features')
    question_features = apply_parallel(_question_features, df.groupby('qid'))
    question_features = list(zip(*question_features))
    question_features = [itertools.chain(*x) for x in question_features]
    # convert generator to list here since it's used multiple times
    question_index = list(question_features[0])
    question_features = question_features[1:]  # skip index
    question_features = [{i: v for i, v in zip(question_index, f)} for f in question_features]
    question_feature_names = [
        'f_question_average_overall_accuracy',
        'f_question_count_total',
        'f_question_count_correct',
        'f_question_count_wrong',
    ]
    for name, feature in zip(question_feature_names, question_features):
        df[name] = df.index.map(feature)

    df['bias'] = 1

    df.to_hdf(featurized_df_dir, 'data')

    return df


def get_split_dfs():
    train_df_dir = os.path.join(DATA_DIR, 'protobowl-042818.log.train.h5')
    test_df_dir = os.path.join(DATA_DIR, 'protobowl-042818.log.test.h5')

    if os.path.exists(train_df_dir) and os.path.exists(test_df_dir):
        log.info('loading train test df')
        return pd.read_hdf(train_df_dir), pd.read_hdf(test_df_dir)

    df = get_featurized_df()

    '''
    def get_first_appearance_date(group):
        group = group.sort_values('date')
        return group.iloc[0]['uid'], group.iloc[0]['date']

    log.info('generating train test df')
    df_by_uid = df.groupby('uid')
    log.info('    order and split users by first appearance dates')
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
    '''

    '''
    # split records by date
    df = df.sort_values('date')
    train_df = df.head(int(len(df) * 0.7))
    test_df = df.tail(int(len(df) * 0.3))
    '''

    # randomly split users
    uids = list(set(df['uid']))
    random.shuffle(uids)
    df_by_uid = df.groupby('uid')
    train_uids = uids[:int(len(uids) * 0.7)]
    test_uids = uids[int(len(uids) * 0.7):]
    train_index = list(itertools.chain(*[
        df_by_uid.get_group(uid).index.tolist() for uid in train_uids]))
    test_index = list(itertools.chain(*[
        df_by_uid.get_group(uid).index.tolist() for uid in test_uids]))
    train_df = df.loc[train_index]
    test_df = df.loc[test_index]

    train_df.to_hdf(train_df_dir, 'data')
    test_df.to_hdf(test_df_dir, 'data')

    return train_df, test_df


def get_split_numpy():
    dirs = [
        os.path.join(DATA_DIR, 'x_train.npy'),
        os.path.join(DATA_DIR, 'y_train.npy'),
        os.path.join(DATA_DIR, 'x_test.npy'),
        os.path.join(DATA_DIR, 'y_test.npy')
    ]
    if all(os.path.exists(d) for d in dirs):
        log.info('loading train test numpy')
        return (np.load(d) for d in dirs)

    train_df, test_df = get_split_dfs()

    log.info('generating train test numpy')
    feature_names = [c for c in train_df.columns if c.startswith('f_')] + ['bias']
    x_train = train_df[feature_names].to_numpy().astype(np.float32)
    y_train = train_df['result_binary'].to_numpy().astype(int)
    x_test = test_df[feature_names].to_numpy().astype(np.float32)
    y_test = test_df['result_binary'].to_numpy().astype(int)

    np.save(dirs[0], x_train)
    np.save(dirs[1], y_train)
    np.save(dirs[2], x_test)
    np.save(dirs[3], y_test)

    return x_train, y_train, x_test, y_test

@DatasetReader.register('retention')
class RetentionReader(DatasetReader):

    def __init__(
            self,
            tokenizer: Tokenizer,
            token_indexers: Dict[str, TokenIndexer],
            max_length: int = 128,
            debug: bool = False,
            lazy: bool = False
    ) -> None:
        super().__init__(lazy)
        self.tokenizer = tokenizer
        self.token_indexers = token_indexers
        self.max_length = max_length
        self.debug = debug

        # embed ids
        self.uid_indexers = {'uid_tokens': SingleIdTokenIndexer(namespace='uid_tokens')}
        self.qid_indexers = {'qid_tokens': SingleIdTokenIndexer(namespace='qid_tokens')}

        # don't re-tokenize text
        self.tokenized_text_cache = {}

    def _read(self, fold: str) -> Iterator[Instance]:
        # precomputed feature vectors
        x_train, y_train, x_test, y_test = get_split_numpy()
        self.mean = np.mean(x_train, axis=0)
        self.std = np.std(x_train, axis=0)
        # don't normalize bias
        self.mean[-1] = 0
        self.std[-1] = 1
        # precompute featurized dataframes
        df_train, df_test = get_split_dfs()
        questions = get_questions()

        if fold == 'train':
            xs, ys = x_train, y_train
            df = df_train
            n_examples = int(len(ys) * 0.1) if self.debug else len(ys)
            xs = xs[:n_examples]
            ys = ys[:n_examples]
        else:
            xs, ys = x_test, y_test
            df = df_test

        for features, y, row in zip(xs, ys, df.iterrows()):
            row = row[1]  # row[0] is index
            text = questions[row.qid]
            yield self.text_to_instance(text, row.uid, row.qid, features, y.item())

    def text_to_instance(
            self,
            text: str,
            uid: str,
            qid: str,
            features: np.ndarray,
            label: Optional[int] = None
    ) -> Instance:
        if qid in self.tokenized_text_cache:
            tokenized_text = self.tokenized_text_cache[qid]
        else:
            tokenized_text = self.tokenizer.tokenize(text)[:self.max_length]
            self.tokenized_text_cache[qid] = tokenized_text
        if len(tokenized_text) == 0:
            return None

        features = (features - self.mean) / self.std

        fields: Dict[str, Field] = {
            'tokens': TextField(tokenized_text, self.token_indexers),
            'uid': TextField([Token(uid)], self.uid_indexers),
            'qid': TextField([Token(qid)], self.qid_indexers),
            'features': ArrayField(features)
        }

        if label is not None:
            fields['label'] = LabelField(label, skip_indexing=True)

        return Instance(fields)
