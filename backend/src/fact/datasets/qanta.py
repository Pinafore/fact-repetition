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

QANTA_TRAIN = 'data/train.qanta.record.json'
QANTA_DEV = 'data/dev.qanta.record.json'
QANTA_TEST = 'data/test.qanta.record.json'

QANTA_QUESTION = 'data/qanta.question.json'

# TODO: Read this from data instead of hard code
# Accuracy of users on questions
AVG_ACCURACY = .614
# Average fraction of question text seen
AVG_FRAC_SEEN = 0.552

@DatasetReader.register('qanta')
class QantaReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        # guesstrain, guessdev, guesstest, buzztrain, buzzdev, buzztest
        # self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter())
        # # TODO: Add character indexer
        # self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path:str):
        with open(file_path) as f:
            record_data = json.load(f)
        with open(QANTA_TRAIN) as f:
            train_record_data = json.load(f)

        df = pd.DataFrame(train_record_data)
        # qid_enc = self.qid_encoding(df)
        # train_qid_list = [[record['qid']] for record in train_record_data]
        # train_qid_feature = qid_enc.transform(train_qid_list)
        accuracy_per_question_feature = self.accuracy_per_question(df)
        # average_buzz_ratio_per_question_feature = self.average_buzz_ratio_per_question(df)
        
        for i in range(len(record_data)):
            # qid_encoding = train_qid_feature[i]
            qid = record_data[i]['qid']
            if qid in accuracy_per_question_feature:
                question_accuracy = accuracy_per_question_feature[qid]
                # question_buzz_ratio = average_buzz_ratio_per_question_feature[qid]
            else:
                question_accuracy = accuracy_per_question_feature['<UKN>']
                # question_buzz_ratio = average_buzz_ratio_per_question_feature['<UKN>']
            label = record_data[i]['ruling']
            instance = self.text_to_instance(
                # text = '',
                # qid_encoding=qid_encoding,
                question_accuracy=question_accuracy,
                # question_buzz_ratio=question_buzz_ratio,
                label=label
                )

        if instance is not None:
            yield instance


    @overrides
    def text_to_instance(self,
                        #  text: str, 
                        # answer: str,
                        #  frac_seen: float = AVG_FRAC_SEEN,
                        #  qid_encoding: np.ndarray,
                         question_accuracy: float,
                        #  question_buzz_ratio: float,
                         # user features
                        #  user_avg_frac_seen: float = AVG_FRAC_SEEN,
                        #  user_avg_accuracy: float = AVG_ACCURACY,
                        #  qanta_id: int = None,
                         label: bool = None):

        fields: Dict[str, Field] = {}
        # tokenized_text = self._tokenizer.tokenize(text)
        # if len(tokenized_text) == 0:
        #     return None
        # fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        # fields['answer'] = LabelField(answer, label_namespace='answer_labels')
        # fields['metadata'] = MetadataField({'qanta_id': qanta_id})
        question_features = [question_accuracy]
        
        fields['question_features'] = ArrayField(np.array(question_features))
        # fields['user_features'] = ArrayField(np.array([user_avg_frac_seen, user_avg_accuracy]))
        if label is not None:
            # fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
            fields['label'] = LabelField(int(label), skip_indexing=True)
        return Instance(fields)

    def accuracy_per_question(self, df):
        avg = df['ruling'].mean()
        accuracy_series = df.groupby(['qid']).mean()['ruling']
        value = []
        for i in range(len(accuracy_series)):
            value.append(accuracy_series[i])
        qid =  list(accuracy_series.keys())
        feature = dict(zip(qid, value))
        feature['<UKN>'] = avg
        return feature

    def average_buzz_ratio_pesr_question(self, df):
        avg = df['buzz_ratio'].mean()
        buzz_ratio_series = df.groupby(['qid']).mean()['buzz_ratio']
        value = []
        for i in range(len(buzz_ratio_series)):
            value.append(buzz_ratio_series[i])
        qid =  list(buzz_ratio_series.keys())
        feature = dict(zip(qid, value))
        feature['<UKN>'] = avg
        return feature

    def qid_encoding(self, df):
        qid_list = list(df.groupby('qid').groups.keys())
        qid_list = [[qid] for qid in qid_list]
        qid_enc = OneHotEncoder(handle_unknown='ignore')
        qid_enc.fit(qid_list)
        return qid_enc