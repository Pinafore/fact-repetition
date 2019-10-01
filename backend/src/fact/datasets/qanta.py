from typing import Dict, List, Union
import json
import pickle
import glob
import math

from tqdm import tqdm
import click
import numpy as np
import pandas as pd
from overrides import overrides
from sklearn.model_selection import train_test_split
from pytorch_pretrained_bert import BertTokenizer

from sklearn.preprocessing import OneHotEncoder
from allennlp.data import DatasetReader, TokenIndexer, Instance
from allennlp.data.fields import TextField, LabelField, Field, MetadataField, ArrayField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, WordSplitter
from allennlp.data.tokenizers.word_stemmer import PassThroughWordStemmer
from allennlp.data.tokenizers.word_filter import PassThroughWordFilter
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

from fact.util import get_logger


log = get_logger(__name__)

TRAIN_RECORD = 'data/train.record.json'
DEV_RECORD = 'data/dev.record.json'
QUESTION_FILE = 'data/avgbert.question.json'

# TODO: Read this from data instead of hard code
# Accuracy of users on questions
# AVG_ACCURACY = .614
# Average fraction of question text seen
# AVG_FRAC_SEEN = 0.552

def bert_embedding(df):
    embedding = df['avg_bert']
    qid =  list(df['qid'])
    feature = dict(zip(qid, embedding))
    feature['<UKN>'] = [0] * len(df['avg_bert'][0])
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

def times_seen(df):
    cumcount_array = df.groupby(['uid', 'qid']).cumcount().values
    feature = [[cumcount] for cumcount in cumcount_array]
    return feature

with open(TRAIN_RECORD) as f:
    train_record = json.load(f)
with open(DEV_RECORD) as f:
    dev_record = json.load(f)
with open(QUESTION_FILE) as f:
    question_data = json.load(f)
train_df = pd.DataFrame(train_record)
dev_df = pd.DataFrame(dev_record)
question_df = pd.DataFrame(question_data)

embedding_dict = bert_embedding(question_df)
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

train_times_seen_feature = times_seen(train_df)
dev_times_seen_feature = times_seen(dev_df)


@DatasetReader.register('qanta')
class QantaReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        # guesstrain, guessdev, guesstest, buzztrain, buzzdev, buzztest
        # self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter())
        # # TODO: Add character indexer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path:str):
        with open(file_path) as f:
            data = json.load(f)

        num = len(data)
        df = pd.DataFrame(data)

        if file_path == 'data/train.record.json': 
            times_seen_feature = train_times_seen_feature
        else:
            times_seen_feature = dev_times_seen_feature

        for i in range(num):
            uid = data[i]['uid']
            qid = data[i]['qid']
            text = question_df.loc[question_df['qid'] == qid]['text'].values
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

            feature_vec = [user_accuracy] + [user_buzzratio] + [uid_count] +  [question_accuracy] + [question_buzzratio] + [qid_count] + times_seen_feature[i]

            label = data[i]['ruling']
            instance = self.text_to_instance(
                # tokens=[Token(word) for word in self._tokenizer(text.keys())],
                tokens = [Token(word) for word in text],
                embedding = embedding,
                feature_vec = feature_vec,
                label=label
                )
            # print("feature_vec", feature_vec)
            # print("feature_vec len", len(feature_vec)
            if file_path == 'data/train.record.json' and i > 2100000:
                break
            if file_path == 'data/dev.record.json' and i > 300000:
                break
            yield instance


    @overrides
    def text_to_instance(self,
                         tokens: List[Token], 
                        # answer: str,
                        #  frac_seen: float = AVG_FRAC_SEEN,
                        #  qid_encoding: np.ndarray,
                         embedding: List[float],
                         feature_vec: List[float],
                        #  question_buzz_ratio: float,
                         # user features
                        #  user_avg_frac_seen: float = AVG_FRAC_SEEN,
                        #  user_avg_accuracy: float = AVG_ACCURACY,
                        #  qanta_id: int = None,
                         label: bool = None):
        sentence_field = TextField(tokens, self._token_indexers)
        fields: Dict[str, Field] = {"tokens": sentence_field}
        # tokenized_text = self._tokenizer.tokenize(text)
        # if len(tokenized_text) == 0:
        #     return None
        # fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        # fields['answer'] = LabelField(answer, label_namespace='answer_labels')
        # fields['metadata'] = MetadataField({'qanta_id': qanta_id})
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

    