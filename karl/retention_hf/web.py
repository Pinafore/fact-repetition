#!/usr/bin/env python
# coding: utf-8

import torch
import logging
import numpy as np
from typing import List
from fastapi import FastAPI

import torch.nn.functional as F
from transformers import DistilBertTokenizerFast, default_data_collator

from karl.retention_hf.model import DistilBertRetentionModel
from karl.retention_hf.data import RetentionFeaturesSchema, RetentionInput, retention_data_collator, feature_fields
from karl.config import settings


class RetentionModel:

    def __init__(self):
        self.model_new_card = DistilBertRetentionModel.from_pretrained(f'{settings.CODE_DIR}/output/retention_hf_new_card')
        self.model_new_card.eval()
        self.model_old_card = DistilBertRetentionModel.from_pretrained(f'{settings.CODE_DIR}/output/retention_hf_old_card')
        self.model_old_card.eval()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.mean = torch.load(f'{settings.DATA_DIR}/cached_mean')
        self.std = torch.load(f'{settings.DATA_DIR}/cached_std')

    def predict(
        self,
        feature_vectors: List[RetentionFeaturesSchema],
    ):
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

        output = [None for _ in feature_vectors]
        if len(new_examples) > 0:
            new_inputs = default_data_collator(new_examples)
            new_output = F.softmax(self.model_new_card.forward(**new_inputs)[0], dim=-1)
            new_output = new_output.detach().cpu().numpy().tolist()
            for i, x in zip(new_indices, new_output):
                output[i] = x
        if len(old_examples) > 0:
            old_inputs = retention_data_collator(old_examples)
            old_output = F.softmax(self.model_old_card.forward(**old_inputs)[0], dim=-1)
            old_output = old_output.detach().cpu().numpy().tolist()
            for i, x in zip(old_indices, old_output):
                output[i] = x
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
