#!/usr/bin/env python
# coding: utf-8

import pytz
import torch
import logging
import numpy as np
from datetime import datetime
from typing import List
from fastapi import FastAPI

from transformers import DistilBertTokenizerFast

from karl.retention_hf import DistilBertRetentionModel
from karl.retention_hf.data import RetentionFeaturesSchema, RetentionInput, retention_data_collator, feature_fields
from karl.config import settings


class RetentionModel:

    def __init__(self):
        model_new_card = DistilBertRetentionModel.from_pretrained(f'{settings.CODE_DIR}/output/retention_hf_distilbert_new_card')
        model_old_card = DistilBertRetentionModel.from_pretrained(f'{settings.CODE_DIR}/output/retention_hf_distilbert_old_card')
        self.model_new_card = model_new_card.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_old_card = model_old_card.to('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_new_card.eval()
        self.model_old_card.eval()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.mean = torch.load(f'{settings.DATA_DIR}/cached_mean')
        self.std = torch.load(f'{settings.DATA_DIR}/cached_std')

    def predict(self, feature_vectors: List[RetentionFeaturesSchema]):
        t0 = datetime.now(pytz.utc)

        card_encodings = self.tokenizer([x.card_text for x in feature_vectors], truncation=True, padding=True)
        new_indices, old_indices = [], []
        new_examples, old_examples = [], []
        for i, x in enumerate(feature_vectors):
            if x.is_new_fact:
                example = {k: v[i] for k, v in card_encodings.items()}
                new_examples.append(RetentionInput(**example))
                new_indices.append(i)
            else:
                retention_features = [x.__dict__[field] for field in feature_fields]
                retention_features = np.array(retention_features)
                retention_features = (retention_features - self.mean) / self.std
                example = {k: v[i] for k, v in card_encodings.items()}
                example['retention_features'] = retention_features
                old_examples.append(RetentionInput(**example))
                old_indices.append(i)

        t1 = datetime.now(pytz.utc)
        print('============ gather inputs', (t1 - t0).total_seconds())

        batch_size = 32
        output = [None for _ in feature_vectors]
        if len(new_examples) > 0:
            for i in range(0, len(new_examples), batch_size):
                xs = retention_data_collator(new_examples[i: i + batch_size])
                if torch.cuda.is_available():
                    xs = {k: v.to('cuda') for k, v in xs.items()}
                ys = torch.sigmoid(self.model_new_card.forward(**xs)[0])
                ys = ys.detach().cpu().numpy().tolist()
                for i, y in zip(new_indices[i: i + batch_size], ys):
                    output[i] = y

        t2 = datetime.now(pytz.utc)
        print('============ predict new', (t2 - t1).total_seconds())

        if len(old_examples) > 0:
            for i in range(0, len(old_examples), batch_size):
                xs = retention_data_collator(old_examples[i: i + batch_size])
                if torch.cuda.is_available():
                    xs = {k: v.to('cuda') for k, v in xs.items()}
                ys = torch.sigmoid(self.model_old_card.forward(**xs)[0])
                ys = ys.detach().cpu().numpy().tolist()
                for i, y in zip(old_indices[i: i + batch_size], ys):
                    output[i] = y

        t3 = datetime.now(pytz.utc)
        print('============ predict old', (t3 - t2).total_seconds())

        return output

    def predict_one(
        self,
        feature_vector: RetentionFeaturesSchema,
    ) -> float:
        return self.predict([feature_vector])


# create logger with 'retention'
logger = logging.getLogger('retention')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler(f'{settings.CODE_DIR}/retention.log')
fh.setLevel(logging.DEBUG)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

app = FastAPI()
retention_model = RetentionModel()


@app.get('/api/karl/predict_one')
def predict_one(feature_vector: RetentionFeaturesSchema):
    return retention_model.predict_one(feature_vector)

@app.get('/api/karl/predict')
def predict(feature_vectors: List[RetentionFeaturesSchema]):
    return retention_model.predict(feature_vectors)
