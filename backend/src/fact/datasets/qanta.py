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

from fact.util import get_logger


log = get_logger(__name__)

QANTA_TRAIN = 'data/expanded.qanta.train.2018.04.18.json'
QANTA_DEV = 'data/expanded.qanta.dev.2018.04.18.json'
QANTA_TEST = 'data/expanded.qanta.test.2018.04.18.json'

USER_HASH = 'data/protobowl_byuser_hash.json'
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
        self._tokenizer = tokenizer or WordTokenizer(SpacyWordSplitter())
        # TODO: Add character indexer
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        #open the dict of by_user statistics
        with open(USER_HASH) as user_file:
            user_hash = json.load(user_file)['data']

        #the following code opens an expanded QANTA file containing questions along with
        #statistics for those questions.  It then sees what user is answering the question 
        #and integrates relevant information for that user.
        with open(file_path) as f:
            questions = json.load(f)['questions']

        for q in questions:
            examples = q['question_stats']
            users = examples['users_per_question']
            fractions_seen = examples['length_per_question']
            user_labels = examples['accuracy_per_question']
            # This assumes that users appear in cronological order:
            question_accuracy = None
            for user_id, frac_seen, label in zip(users, fractions_seen, user_labels):
                user_data = user_hash[user_id]
                # logic to calculate how much of question the user had seen
                text_idx =  math.floor(len(q['text'] ) * frac_seen)
                text_seen = q['text'][:text_idx]

                # This should be done outside of here, but as is the data is not computed correctly
                # since it does not compute stats excluding the current question.
                # In the worst case, the user has a single question (the current one) and the features leak
                # the label itself
                user_features = {'frac_seen': [], 'label': []}
                for qid, user_frac_seen, user_label in zip(user_data['questions_per_user'], user_data['accuracy_per_user'], user_data['length_per_user']):
                    if qid != q['proto_id']:
                        user_features['frac_seen'].append(user_frac_seen)
                        user_features['label'].append(int(user_label))

                if len(user_features['frac_seen']) == 0:
                    avg_user_accuracy = AVG_ACCURACY
                    avg_user_frac_seen = AVG_FRAC_SEEN
                else:
                    user_avg_frac_seen = np.mean(user_features['frac_seen'])
                    user_avg_accuracy = np.mean(user_features['label'])
                instance = self.text_to_instance(
                    text_seen, q['page'],
                    qanta_id=q['qanta_id'],
                    #question features
                    frac_seen=frac_seen,
                    question_accuracy = AVG_ACCURACY if question_accuracy is None else np.mean(question_accuracy),

                    #user features
                    user_avg_frac_seen=user_avg_frac_seen,
                    user_avg_accuracy=user_avg_accuracy,
                    label=label
                )
                if question_accuracy is None:
                    question_accuracy = []
                question_accuracy.append(int(label))

                if instance is not None:
                    yield instance
    @overrides
    def text_to_instance(self, text: str, answer: str,
                         frac_seen: float = AVG_FRAC_SEEN,
                         question_accuracy: float = AVG_ACCURACY,

                         # user features
                         user_avg_frac_seen: float = AVG_FRAC_SEEN,
                         user_avg_accuracy: float = AVG_ACCURACY,
                         qanta_id: int = None,
                         label: bool = None):

        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        if len(tokenized_text) == 0:
            return None
        fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        fields['answer'] = LabelField(answer, label_namespace='answer_labels')
        fields['metadata'] = MetadataField({'qanta_id': qanta_id})
        fields['question_features'] = ArrayField(np.array([frac_seen, question_accuracy]))
        fields['user_features'] = ArrayField(np.array([user_avg_frac_seen, user_avg_accuracy]))
        if label is not None:
            fields['label'] = LabelField(int(label), skip_indexing=True)
        return Instance(fields)
