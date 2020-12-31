#!/usr/bin/env python
# coding: utf-8

import torch
import logging
import numpy as np
from typing import List
from fastapi import FastAPI
import torch.nn.functional as F

from karl.retention_hf.model import DistilBertRetentionModel
from karl.retention_hf.data import RetentionDataset, RetentionFeaturesSchema, RetentionInput
from karl.config import settings


class RetentionModel:

    def __init__(self):
        self.model_new_card = DistilBertRetentionModel.from_pretrained(f'{settings.CODE_DIR}/output/retention_hf_new_card')
        self.model_new_card.eval()
        self.model_old_card = DistilBertRetentionModel.from_pretrained(f'{settings.CODE_DIR}/output/retention_hf_old_card')
        self.model_old_card.eval()
        self.tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
        self.dataset = RetentionDataset()
        self.feature_fields = [
            field_name for field_name, field_info in RetentionFeaturesSchema.__fields__.items()
            if field_info.type_ in [int, float, bool]
        ]

    def predict_one(
        self,
        feature_vector: RetentionFeaturesSchema,
    ) -> float:
        card_encodings = self.tokenizer(card_texts, truncation=True, padding=True)
        if feature_vector.is_new_fact:
            example = RetentionInput(**card_encodings)
        else:
            retention_features = [feature_vector.__dict__[field] for field in self.feature_fields]
            retention_features = np.array(retention_features)
            retention_features = (retention_features - self.dataset.mean) / self.dataset.std
            example = card_encodings
            example['retention_features'] = retention_features
        return self.model(RetentionInput(**example))

    def predict(
        self,
        feature_vectors: List[RetentionFeaturesSchema],
    ) -> List[float]:
        card_encodings = self.tokenizer(card_texts, truncation=True, padding=True)
        if feature_vector.is_new_fact:
            example = RetentionInput(**card_encodings)
        else:
            retention_features = [feature_vector.__dict__[field] for field in self.feature_fields]
            retention_features = np.array(retention_features)
            retention_features = (retention_features - self.dataset.mean) / self.dataset.std
            example = card_encodings
            example['retention_features'] = retention_features
        return self.model(RetentionInput(**example))


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
