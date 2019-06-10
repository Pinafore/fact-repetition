from typing import Dict, List, Union
import json
import pickle
import glob

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

QANTA_TRAIN = 'data/qanta.train.2018.04.18.json'
QANTA_DEV = 'data/qanta.dev.2018.04.18.json'
QANTA_TEST = 'data/qanta.test.2018.04.18.json'


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
        with open(file_path) as f:
            for q in json.load(f)['questions']:
                if q['page'] is not None and q['fold'] == self._fold:
                    if self._break_questions:
                        for start, end in q['tokenizations']:
                            sentence = q['text'][start:end]
                            instance = self.text_to_instance(sentence, answer=q['page'], qanta_id=q['qanta_id'])
                            if instance is not None:
                                yield instance
                    else:
                        instance = self.text_to_instance(q['text'], answer=q['page'], qanta_id=q['qanta_id'])
                        if instance is not None:
                            yield instance

    @overrides
    def text_to_instance(self,
                         text: str,
                         answer: str = None,
                         qanta_id: int = None):
        fields: Dict[str, Field] = {}
        tokenized_text = self._tokenizer.tokenize(text)
        if len(tokenized_text) == 0:
            return None
        fields['text'] = TextField(tokenized_text, token_indexers=self._token_indexers)
        if answer is not None:
            fields['answer'] = LabelField(answer, label_namespace='answer_labels')
        fields['metadata'] = MetadataField({'qanta_id': qanta_id})
        return Instance(fields)