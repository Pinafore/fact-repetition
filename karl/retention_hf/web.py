#!/usr/bin/env python
# coding: utf-8

import torch
import logging
import numpy as np
from typing import List
from fastapi import FastAPI

from transformers import DistilBertTokenizerFast, default_data_collator

from karl.retention_hf.model import DistilBertRetentionModel
from karl.retention_hf.data import RetentionFeaturesSchema, RetentionInput, retention_data_collator
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
        self.feature_fields = [
            field_name for field_name, field_info in RetentionFeaturesSchema.__fields__.items()
            if field_info.type_ in [int, float, bool]
        ]

    def predict(
        self,
        feature_vectors: List[RetentionFeaturesSchema],
    ):
        card_encodings = self.tokenizer([x.card_text for x in feature_vectors], truncation=True, padding=True)

        # TODO
        if feature_vectors[0].is_new_fact:
            examples = []
            for i, _ in enumerate(feature_vectors):
                example = {k: v[i] for k, v in card_encodings.items()}
                examples.append(RetentionInput(**example))
            inputs = default_data_collator(examples)
            output = self.model_new_card.forward(**inputs)[0].detach().cpu().numpy().tolist()
        else:
            examples = []
            for i, x in enumerate(feature_vectors):
                retention_features = [x.__dict__[field] for field in self.feature_fields]
                retention_features = np.array(retention_features)
                retention_features = (retention_features - self.mean) / self.std
                example = {k: v[i] for k, v in card_encodings.items()}
                example['retention_features'] = retention_features
                examples.append(RetentionInput(**example))
            inputs = retention_data_collator(examples)
            output = self.model_old_card.forward(**inputs)[0].detach().cpu().numpy().tolist()
        # TODO check for NaN
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
