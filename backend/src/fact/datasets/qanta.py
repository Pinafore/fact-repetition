from typing import Dict, List, Union
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
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from fact.util import get_logger


log = get_logger(__name__)

TRAIN_RECORD = 'data/train.record.json'
DEV_RECORD = 'data/dev.record.json'
QUESTION_FILE = 'data/avgbert.question.json'
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

def bert_embedding(question_data):
    n_features = 0
    feature = {}
    for qid, q in question_data.items():
        if n_features == 0:
            n_features = len(q['avg_bert'])
        feature[qid] = q['avg_bert']
    feature['<UKN>'] = [0] * n_features
    return feature

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

def uid_encoding(df):
    uid_list = list(df.groupby('uid').groups.keys())
    uid_list = [[uid] for uid in uid_list]
    uid_enc = OneHotEncoder(handle_unknown='ignore')
    uid_enc.fit(uid_list)
    return uid_enc

def qid_encoding(df):
    qid_list = list(df.groupby('qid').groups.keys())
    qid_list = [[qid] for qid in qid_list]
    qid_enc = OneHotEncoder(handle_unknown='ignore')
    qid_enc.fit(qid_list)
    return qid_enc

def uid_count(df):
    uid = list(df.groupby('uid').groups.keys())
    uid_count_series = df.groupby(['uid']).mean()['uid_count']
    value = []
    for i in range(len(uid_count_series)):
        value.append(uid_count_series[i])
    feature = dict(zip(uid, value))
    feature['<UKN>'] = 0
    return feature

def qid_count(df):
    qid = list(df.groupby('qid').groups.keys())
    qid_count_series = df.groupby(['qid']).mean()['qid_count']
    value = []
    for i in range(len(qid_count_series)):
        value.append(qid_count_series[i])
    feature = dict(zip(qid, value))
    feature['<UKN>'] = 0
    return feature

# def times_seen(df):
#     ruling_df = pd.DataFrame({'count' : df.groupby(['uid', 'qid', 'ruling']).size()}).reset_index()
#     ruling_count_df = pd.pivot_table(ruling_df, values='count', index=['uid', 'qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
#     feature = {}
#     for i in range(len(ruling_count_df.index)):
#         feature[ruling_count_df.index[i]] = ruling_count_df[0][i] + ruling_count_df[1][i]
#     return feature

# def times_seen_correct(df):
#     ruling_df = pd.DataFrame({'count' : df.groupby(['uid', 'qid', 'ruling']).size()}).reset_index()
#     ruling_count_df = pd.pivot_table(ruling_df, values='count', index=['uid', 'qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
#     feature = {}
#     for i in range(len(ruling_count_df.index)):
#         feature[ruling_count_df.index[i]] = ruling_count_df[1][i] # "False  True"
#     return feature

# def times_seen_wrong(df):
#     ruling_df = pd.DataFrame({'count' : df.groupby(['uid', 'qid', 'ruling']).size()}).reset_index()
#     ruling_count_df = pd.pivot_table(ruling_df, values='count', index=['uid', 'qid'], columns=['ruling'], aggfunc=np.sum).fillna(0)
#     feature = {}
#     for i in range(len(ruling_count_df.index)):
#         feature[ruling_count_df.index[i]] = ruling_count_df[0][i]
#     return feature


with open(TRAIN_RECORD) as f:
    train_record = json.load(f)
with open(DEV_RECORD) as f:
    dev_record = json.load(f)
with open(QUESTION_FILE) as f:
    question_data = json.load(f)
    question_data = {q['qid']: q for q in question_data}
with open(TIMES_SEEN) as f:
    times_seen_feature = json.load(f)
with open(TIMES_SEEN_CORRECT) as f:
    times_seen_correct_feature = json.load(f)
with open(TIMES_SEEN_WRONG) as f:
    times_seen_wrong_feature = json.load(f)
train_df = pd.DataFrame(train_record)
dev_df = pd.DataFrame(dev_record)
question_df = pd.DataFrame(question_data)

embedding_dict = bert_embedding(question_data)
accuracy_per_user_feature = accuracy_per_user(train_df)
accuracy_per_question_feature = accuracy_per_question(train_df)
average_buzz_ratio_per_user_feature = average_buzz_ratio_per_user(train_df)
average_buzz_ratio_per_question_feature = average_buzz_ratio_per_question(train_df)

# qid_enc = qid_encoding(train_df)
# uid_enc = uid_encoding(train_df)
# uid_list = [[record['uid']] for record in train_record]
# qid_list = [[record['qid']] for record in train_record]
# uid_feature = uid_enc.transform(uid_list)
# qid_feature = qid_enc.transform(qid_list)

uid_count_feature = uid_count(train_df)
qid_count_feature = qid_count(train_df)


@DatasetReader.register('qanta')
class QantaReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        # guesstrain, guessdev, guesstest, buzztrain, buzzdev, buzztest
        # self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        # self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter())
        self._spacetokenizer = WordTokenizer(JustSpacesWordSplitter())
        self._token_uid_indexers = {'uid_tokens': SingleIdTokenIndexer(namespace='uid_tokens')}
        self._token_qid_indexers = {'qid_tokens': SingleIdTokenIndexer(namespace='qid_tokens')}
        # # # TODO: Add character indexer
        # self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._token_indexers = {'tokens': SingleIdTokenIndexer(namespace='tokens')}

    @overrides
    def _read(self, file_path:str):
        with open(file_path) as f:
            data = json.load(f)

        num = len(data)

        # if file_path == 'data/train.record.json': 
        #     times_seen_feature = train_times_seen_feature
        # else:
        #     times_seen_feature = dev_times_seen_feature

        for i in range(num):
            uid = data[i]['uid']
            qid = data[i]['qid']
            text = question_data[qid]['text']
            # print("uid", uid)
            # print("qid", qid)
            # print("text", text)
            #
            if uid in accuracy_per_user_feature:
                user_accuracy = accuracy_per_user_feature[uid]
                user_buzzratio = average_buzz_ratio_per_user_feature[uid]
                uid_count = uid_count_feature[uid]
            else:
                user_accuracy = accuracy_per_user_feature['<UKN>']
                user_buzzratio = average_buzz_ratio_per_user_feature['<UKN>']
                uid_count = uid_count_feature['<UKN>']
            if qid in accuracy_per_question_feature:
                embedding = embedding_dict[qid]
                question_accuracy = accuracy_per_question_feature[qid]
                question_buzzratio = average_buzz_ratio_per_question_feature[qid]
                qid_count = qid_count_feature[qid]
            else:
                embedding = embedding_dict['<UKN>']
                question_accuracy = accuracy_per_question_feature['<UKN>']
                question_buzzratio = average_buzz_ratio_per_question_feature['<UKN>']
                qid_count = qid_count_feature['<UKN>']

            if str((uid, qid)) in times_seen_feature:
                times_seen = times_seen_feature[str((uid, qid))]
                times_seen_correct = times_seen_correct_feature[str((uid, qid))]
                times_seen_wrong = times_seen_wrong_feature[str((uid, qid))]
            else:
                times_seen = 0
                times_seen_correct = 0
                times_seen_wrong = 0

            feature_vec = [user_accuracy] + [user_buzzratio] + [uid_count] + [question_accuracy] + [question_buzzratio] + [qid_count] + [times_seen] + [times_seen_correct] + [times_seen_wrong]

            label = data[i]['ruling']
            # text = text[0:1]
            instance = self.text_to_instance(
                tokens = self._spacetokenizer.tokenize(text)[:150],
                # uid_text = [Token(uid)],
                # qid_text = [Token(qid)],
                uid_tokens = self._spacetokenizer.tokenize(uid),
                qid_tokens = self._spacetokenizer.tokenize(qid),
                embedding = embedding,
                feature_vec = feature_vec,
                label=label
                )
            # print("tokens", self._spacetokenizer.tokenize(text))
            # print("qid_tokens", self._spacetokenizer.tokenize(uid))
            # print("qid_tokens", self._spacetokenizer.tokenize(qid))
            # if i % 1000 == 0:
            #     print(i, "000 records.")
            if file_path == 'data/train.record.json' and i > 693:
                break
            if file_path == 'data/dev.record.json' and i > 99:
                break
            yield instance
        
        self._tokenizer.save("./fine_tuned_bert_model")


    @overrides
    def text_to_instance(self,
                         tokens: List[Token], 
                        # answer: str,
                        #  frac_seen: float = AVG_FRAC_SEEN,
                        #  qid_encoding: np.ndarray,
                         uid_tokens: str,
                         qid_tokens: str,
                         embedding: List[float],
                         feature_vec: List[float],
                        #  question_buzz_ratio: float,
                        # user features
                        #  user_avg_frac_seen: float = AVG_FRAC_SEEN,
                        #  user_avg_accuracy: float = AVG_ACCURACY,
                        #  qanta_id: int = None,
                         label: bool = None):
        fields = {}
        fields['uid_tokens'] = TextField(uid_tokens, self._token_uid_indexers)
        fields['qid_tokens'] = TextField(qid_tokens, self._token_qid_indexers)
        fields['tokens'] = TextField(tokens, self._token_indexers)
        # tokenized_text = self._tokenizer.tokenize(text)
        # if len(tokenized_text) == 0:
        #     return None
        # fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        # fields['answer'] = LabelField(answer, label_namespace='answer_labels')
        # fields['metadata'] = MetadataField({'qanta_id': qanta_id})
        # print("tokens", fields['tokens'])
        # print("uid_tokens", fields['uid_tokens'])
        # print("qid_tokens", fields['qid_tokens'])
        fields['embedding'] = ArrayField(np.array(embedding))
        fields['feature_vec'] = ArrayField(np.array(feature_vec))
        # fields['user_features'] = ArrayField(np.array([user_avg_frac_seen, user_avg_accuracy]))
        if label is not None:
            # fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
            fields['label'] = LabelField(int(label), skip_indexing=True)
        return Instance(fields)

    # def category_subcategory_difficulty(self, question_df):
    #     df = question_df
    #     print("question len: ", len(df.groupby('qid')))
    #     enc = OneHotEncoder(handle_unknown='ignore')
    #     category_list = list(df.groupby('category').groups.keys())
    #     # subcategory_list = list(df.groupby('subcategory').groups.keys())
    #     difficulty_list = list(df.groupby('difficulty').groups.keys())
    #     category_subcategory_difficulty_list = []
    #     sorted_len = [len(category_list), len(subcategory_list), len(difficulty_list)]
    #     sorted_len.sort()
    #     for i in range(sorted_len[-1]):
    #         if i < sorted_len[0]:
    #             category_subcategory_difficulty_list.append([category_list[i], subcategory_list[i], difficulty_list[i]])
    #         elif i < sorted_len[1]:
    #             category_subcategory_difficulty_list.append([category_list[i], subcategory_list[i], difficulty_list[0]])
    #         else:
    #             category_subcategory_difficulty_list.append([category_list[0], subcategory_list[i], difficulty_list[0]])
    #     enc.fit(category_subcategory_difficulty_list)
    #     return enc

    