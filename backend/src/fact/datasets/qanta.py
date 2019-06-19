from typing import Dict, List, Union
import json
import pickle
import glob
import math

from tqdm import tqdm
import click
import numpy as np
from overrides import overrides
from sklearn.model_selection import train_test_split

from allennlp.data import DatasetReader, TokenIndexer, Instance
from allennlp.data.fields import TextField, LabelField, Field, MetadataField, ArrayField, ListField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer, Token
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, WordSplitter
from allennlp.data.tokenizers.word_stemmer import PassThroughWordStemmer
from allennlp.data.tokenizers.word_filter import PassThroughWordFilter
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer

#from fact.util import get_logger


#log = get_logger(__name__)

QANTA_TRAIN = 'data/expanded.qanta.train.2018.04.18.json'
QANTA_DEV = 'data/expanded.qanta.dev.2018.04.18.json'
QANTA_TEST = 'data/expanded.qanta.test.2018.04.18.json'

USER_HASH = 'data/protobowl_byuser_hash'

@DatasetReader.register('qanta')
class QantaReader(DatasetReader):
    def __init__(self,
                 fold: str,
                 break_questions: bool,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        # guesstrain, guessdev, guesstest, buzztrain, buzzdev, buzztest
        self._fold = fold
        self._break_questions = break_questions
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter())
        # TODO: Add character indexer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        #open the dict of by_user statistics
        with open(USER_HASH) as user_file:
            user_hash = json.load(user_file)['data']

        with open(file_path) as f:
            for q in json.load(f)['questions']:
                relevant_users = q['questions_stats']['users_per_question']
                for user in relevant_users:
                    user_data = user_hash[user]

                    if q['page'] is not None and q['fold'] == self._fold:

                        #logic to calculate how much of question the user had seen
                        index_of_user = q['question_stats']['users_per_questions'].index(user)
                        question_percent_seen = q['question_stats']['length_per_question'][index_of_user]
                        index =  math.floor(len(q['text'] ) * question_percent_seen)

                        seen_data = q['text'][0:index]
                        instance = self.text_to_instance(seen_data, answer=q['page'], qanta_id=q['qanta_id'],
                                                             #question features
                                                             length_per_question = q['question_stats']['length_per_question'],
                                                             overall_length_per_question = q['question_stats']['overall_length_per_question'],
                                                             overall_accuracy_per_question = q['question_stats']['overall_accuracy_per_question'],
                                                             accuracy_per_question = q['question_stats']['accuracy_per_question'],

                                                             #user features
                                                             user_length_ratios = user_data['length_per_user'],
                                                             overall_length = user_data['overall_length_per_user'],
                                                             accuracy_per_user = user_data['accuracy_per_user'],
                                                             overall_accuracy_per_user = user_data['overall_accuracy_per_user']
                                                             )
                        if instance is not None:
                             yield instance
    @overrides
    def text_to_instance(self,
                         text: str,
                         answer: str = None,
                         qanta_id: int = None,
                         length_per_question: List[float] = None,
                         overall_length_per_question: float = None,
                         accuracy_per_question: float= None,
                         overall_accuracy_per_question: List[bool] = None,

                         # user features
                         length_per_user: float = None,
                         overall_length_per_user: List[float] = None,
                         accuracy_per_user: List[bool] = None,
                         overall_accuracy_per_user: bool = None
                        ):

        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        if len(tokenized_text) == 0:
            return None
        fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        if answer is not None:
            fields['answer'] = LabelField(answer, label_namespace='answer_labels')

        fields['metadata'] = MetadataField({'qanta_id': qanta_id})

        field['question_features'] = ArrayField({'length_per_question':length_per_question,
                                            'overall_length_per_question': overall_length_per_question,
                                           'overall_accuracy_per_question':overall_accuracy_per_question,
                                           'individual_accuracy_per_question':accuracy_per_question})

        field['user_features'] = ArrayField({'length_per_user': length_per_user,
                                            'overall_length_per_user': overall_length_per_user,
                                             'accuracy_per_user':accuracy_per_user,
                                            'overall accuracy_per_user': overall_accuracy_per_user
                                           })

        return Instance(fields)