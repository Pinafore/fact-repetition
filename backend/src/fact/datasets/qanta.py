from typing import Dict, List, Union, Optional
import json
import pickle
import glob
import math

from tqdm import tqdm
import click
import numpy as np
import pandas as pd
import torch
from overrides import overrides
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer

from sklearn.preprocessing import OneHotEncoder
from allennlp.data import DatasetReader, TokenIndexer, Instance
from allennlp.data.fields import TextField, LabelField, Field, MetadataField, ArrayField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, WordSplitter, JustSpacesWordSplitter
from allennlp.data.tokenizers.word_stemmer import PassThroughWordStemmer
from allennlp.data.tokenizers.word_filter import PassThroughWordFilter

from fact.util import get_logger


log = get_logger(__name__)

TRAIN_RECORD = 'data/train.record.json'
DEV_RECORD = 'data/dev.record.json'
QUESTION_FILE = 'data/onlytext.question.json'
QID_FILE = 'data/qid_array.txt'
UID_FILE = 'data/uid_array.txt'
TIMES_SEEN = 'data/times_seen.json'
TIMES_SEEN_CORRECT = 'data/times_seen_correct.json'
TIMES_SEEN_WRONG = 'data/times_seen_wrong.json'

# TODO: Read this from data instead of hard code
# Accuracy of users on questions
# AVG_ACCURACY = .614
# Average fraction of question text seen
# AVG_FRAC_SEEN = 0.552

def get_revealed_text(text, buzz_ratio):
    length = math.floor(len(text) * buzz_ratio)
    return text[0:length]

def accuracy_per_user(df):
    avg = df['ruling'].mean()
    accuracy_series = df.groupby(['uid']).mean()['ruling']
    value = []
    for i in range(len(accuracy_series)):
        value.append(accuracy_series[i])
    uid =  list(accuracy_series.keys())
    feature = dict(zip(uid, value))
    feature['<UKN>'] = avg
    return feature

def average_buzz_ratio_per_user(df):
    avg = df['buzz_ratio'].mean()
    buzz_ratio_series = df.groupby(['uid']).mean()['buzz_ratio']
    value = []
    for i in range(len(buzz_ratio_series)):
        value.append(buzz_ratio_series[i])
    uid =  list(buzz_ratio_series.keys())
    feature = dict(zip(uid, value))
    feature['<UKN>'] = avg
    return feature

def accuracy_per_question(df):
    avg = df['ruling'].mean()
    accuracy_series = df.groupby(['qid']).mean()['ruling']
    value = []
    for i in range(len(accuracy_series)):
        value.append(accuracy_series[i])
    qid =  list(accuracy_series.keys())
    feature = dict(zip(qid, value))
    feature['<UKN>'] = avg
    return feature

def average_buzz_ratio_per_question(df):
    avg = df['buzz_ratio'].mean()
    buzz_ratio_series = df.groupby(['qid']).mean()['buzz_ratio']
    value = []
    for i in range(len(buzz_ratio_series)):
        value.append(buzz_ratio_series[i])
    qid =  list(buzz_ratio_series.keys())
    feature = dict(zip(qid, value))
    feature['<UKN>'] = avg
    return feature

def uid_count(df):
    uid = list(df.groupby('uid').groups.keys())
    uid_count_series = df.groupby(['uid']).size().reset_index(name='uid_count')['uid_count']
    value = []
    for i in range(len(uid_count_series)):
        value.append(uid_count_series[i])
    feature = dict(zip(uid, value))
    feature['<UKN>'] = 0
    return feature

def qid_count(df):
    qid = list(df.groupby('qid').groups.keys())
    qid_count_series = df.groupby(['qid']).size().reset_index(name='qid_count')['qid_count']
    value = []
    for i in range(len(qid_count_series)):
        value.append(qid_count_series[i])
    feature = dict(zip(qid, value))
    feature['<UKN>'] = 0
    return feature

def category(question_df):
    category_list = question_df.groupby(['category']).size().keys().values[1:] #contain Unlabeled, remove noise
    feature = {}
    num = len(category_list) # 35
    for category in category_list:
        feature[category] = [0] * num
        feature[category][np.where(category_list == category)[0][0]] = 1
    return feature

def difficulty(question_df):
    difficulty_list = question_df.groupby(['difficulty']).size().keys().values[:-1] #remove noise
    difficulty_list = np.append(difficulty_list, 'Unlabeled') # 19
    feature = {}
    num = len(difficulty_list)
    for difficulty in difficulty_list:
        feature[difficulty] = [0] * num
        feature[difficulty][np.where(difficulty_list == difficulty)[0][0]] = 1
    return feature

@DatasetReader.register('qanta')
class QantaReader(DatasetReader):
    def __init__(self,
                 use_bert: bool,
                 use_rnn: bool,
                 tokenizer: Tokenizer,
                 token_indexers: Dict[str, TokenIndexer],
                 lazy: bool = False):
        super().__init__(lazy)
        # guesstrain, guessdev, guesstest, buzztrain, buzzdev, buzztest
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self._uid_indexers = {'uid_tokens': SingleIdTokenIndexer(namespace='uid_tokens')}
        self._qid_indexers = {'qid_tokens': SingleIdTokenIndexer(namespace='qid_tokens')}
        log.info('Reading aux data')
        log.info('train records')
        with open(TRAIN_RECORD) as f:
            train_record = json.load(f)

        log.info('dev records')
        with open(DEV_RECORD) as f:
            dev_record = json.load(f)

        log.info('question file')
        with open(QUESTION_FILE) as f:
            question_data = json.load(f)
            self._question_data = {q['qid']: q for q in question_data}

        log.info('times seen data')
        with open(TIMES_SEEN) as f:
            self._times_seen_feature = json.load(f)
        with open(TIMES_SEEN_CORRECT) as f:
            self._times_seen_correct_feature = json.load(f)
        with open(TIMES_SEEN_WRONG) as f:
            self._times_seen_wrong_feature = json.load(f)
        train_df = pd.DataFrame(train_record)
        dev_df = pd.DataFrame(dev_record)
        question_df = pd.DataFrame(question_data)

        log.info('Computing features')
        self._accuracy_per_user_feature = accuracy_per_user(train_df)
        self._accuracy_per_question_feature = accuracy_per_question(train_df)
        self._average_buzz_ratio_per_user_feature = average_buzz_ratio_per_user(train_df)
        self._average_buzz_ratio_per_question_feature = average_buzz_ratio_per_question(train_df)

        self._uid_count_feature = uid_count(train_df)
        self._qid_count_feature = qid_count(train_df)

        self._category_feature = category(question_df)
        self._difficulty_feature = difficulty(question_df)

        self._question_cache = {}

    @overrides
    def _read(self, file_path:str):
        with open(file_path) as f:
            data = json.load(f)
        i = 0
        print("==========", file_path)
        for row in data:
            i += 1
            uid = row['uid']
            qid = row['qid']
            text = self._question_data[qid]['text']
            label = row['ruling']
            text = get_revealed_text(text, row['buzz_ratio'])
            # print("=========text==========", text)
            instance = self.text_to_instance(
                text=text,
                user_id=uid,
                question_id=qid,
                label='correct' if label else 'wrong'
            )
            if file_path == 'data/train.record.json' and i > 945000:
                break
            if file_path == 'data/dev.record.json' and i > 135000:
                break

            if instance is None:
                continue
            else:
                yield instance

    @overrides
    def text_to_instance(self,
                         text: str, 
                        # answer: str,
                         user_id: str,
                         question_id: str,
                         user_accuracy: Optional[float] = None,
                         user_buzzratio: Optional[float] = None,
                         user_count: Optional[float] = None,
                         question_accuracy: Optional[float] = None,
                         question_buzzratio: Optional[float] = None,
                         question_count: Optional[float] = None,
                         times_seen: Optional[float] = None,
                         times_seen_correct: Optional[float] = None,
                         times_seen_wrong: Optional[float] = None,
                         category: Optional[str] = None,
                         difficulty: Optional[str] = None,
                         label: str = None):
        fields = {}
        if text in self._question_cache:
            tokenized_text = self._question_cache[text]
        else:
            tokenized_text = self._tokenizer.tokenize(text)[:150]
            self._question_cache[text] = tokenized_text
        if len(tokenized_text) == 0:
            return None

        uid = user_id
        qid = question_id

        if user_accuracy is None:
            if uid in self._accuracy_per_user_feature:
                user_accuracy = self._accuracy_per_user_feature[uid]
                user_buzzratio = self._average_buzz_ratio_per_user_feature[uid]
                uid_count = self._uid_count_feature[uid]
            else:
                user_accuracy = self._accuracy_per_user_feature['<UKN>']
                user_buzzratio = self._average_buzz_ratio_per_user_feature['<UKN>']
                uid_count = self._uid_count_feature['<UKN>']

        if question_accuracy is None:
            if qid in self._accuracy_per_question_feature:
                question_accuracy = self._accuracy_per_question_feature[qid]
                question_buzzratio = self._average_buzz_ratio_per_question_feature[qid]
                qid_count = self._qid_count_feature[qid]
            else:
                question_accuracy = self._accuracy_per_question_feature['<UKN>']
                question_buzzratio = self._average_buzz_ratio_per_question_feature['<UKN>']
                qid_count = self._qid_count_feature['<UKN>']

        if times_seen is None:
            paired_id = str((uid, qid))
            if paired_id in self._times_seen_feature:
                times_seen = self._times_seen_feature[paired_id]
                times_seen_correct = self._times_seen_correct_feature[paired_id]
                times_seen_wrong = self._times_seen_wrong_feature[paired_id]
            else:
                times_seen = 0
                times_seen_correct = 0
                times_seen_wrong = 0

        if category is None:
            category = self._question_data[qid]['category']
            if category in self._category_feature:
                category = self._category_feature[category]
            else:
                category = self._category_feature['Unlabeled']
        
        if difficulty is None:
            difficulty = self._question_data[qid]['difficulty']
            if difficulty in self._difficulty_feature:
                difficulty = self._difficulty_feature[difficulty]
            else:
                difficulty = self._difficulty_feature['Unlabeled']

        feature_vec = [
            user_accuracy, user_buzzratio,
            question_accuracy, question_buzzratio,
            uid_count, qid_count,
            times_seen, times_seen_correct, times_seen_wrong,
        ]

        feature_vec = np.concatenate((feature_vec, category, difficulty), axis=None)

        fields['user_id'] = TextField([Token(user_id)], self._uid_indexers)
        fields['question_id'] = TextField([Token(question_id)], self._qid_indexers)
        fields['tokens'] = TextField(tokenized_text, self._token_indexers)
        # fields['metadata'] = MetadataField({'qanta_id': qanta_id})
        fields['feature_vec'] = ArrayField(feature_vec)
        # fields['user_features'] = ArrayField(np.array([user_avg_frac_seen, user_avg_accuracy]))
        if label is not None:
            # fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
            fields['label'] = LabelField(label)
        return Instance(fields)