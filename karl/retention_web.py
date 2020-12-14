#!/usr/bin/env python
# coding: utf-8

import json
import logging
from typing import Generator
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Depends
from dateutil.parser import parse as parse_date
from cachetools import cached, TTLCache
from concurrent.futures import ProcessPoolExecutor
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

import argparse
import numpy as np
from typing import List
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

from karl.models import User, Fact
from karl.retention.data import RetentionDataset, apply_parallel, get_split_dfs
from karl.retention.baseline import TemperatureScaledNet


class RetentionFeatures(BaseModel):
    user_count_correct: float
    user_count_wrong: float
    user_count_total: float
    user_average_overall_accuracy: float
    user_average_question_accuracy: float
    user_previous_result: float
    user_gap_from_previous: float
    question_average_overall_accuracy: float
    question_count_total: float
    question_count_correct: float
    question_count_wrong: float


class RetentionModel:

    def __init__(
        self,
        use_cuda=True,
        checkpoint_dir='checkpoints/retention_model.pt',
    ):
        self.dataset = RetentionDataset()
        n_input = self.dataset.x.shape[1]
        use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model = TemperatureScaledNet(n_input=n_input).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_dir))
        self.model.eval()

    def compute_features(
        self,
        user: User,
        fact: Fact,
        date: datetime,
    ):
        uq_correct = user.count_correct_before.get(fact.fact_id, 0)
        uq_wrong = user.count_wrong_before.get(fact.fact_id, 0)
        uq_total = uq_correct + uq_wrong
        if fact.fact_id in user.previous_study:
            prev_date, prev_response = user.previous_study[fact.fact_id]
        else:
            # TODO this really shouldn't be the current date.
            # the default prev_date should be something much earlier
            # prev_date = str(date)
            prev_response = False
            prev_date = str(date - timedelta(days=10))
            # prev_date = '2020-06-01'
        prev_date = parse_date(prev_date)

        features = [
            uq_correct,  # user_count_correct
            uq_wrong,  # user_count_wrong
            uq_total,  # user_count_total
            0 if len(user.results) == 0 else np.mean(user.results),  # user_average_overall_accuracy
            0 if uq_total == 0 else uq_correct / uq_total,  # user_average_question_accuracy
            prev_response,
            (date - prev_date).total_seconds() / (60 * 60),  # user_gap_from_previous
            0 if len(fact.results) == 0 else np.mean(fact.results),  # question_average_overall_accuracy
            len(fact.results),  # question_count_total
            sum(fact.results),  # question_count_correct
            len(fact.results) - sum(fact.results),  # question_count_wrong
            1  # bias
        ]
        feature_names = [
            'user_count_correct',
            'user_count_wrong',
            'user_count_total',
            'user_average_overall_accuracy',
            'user_average_question_accuracy',
            'user_previous_result',
            'user_gap_from_previous',
            'question_average_overall_accuracy',
            'question_count_total',
            'question_count_correct',
            'question_count_wrong',
            'bias',
        ]
        feature_dict = {k: v for k, v in zip(feature_names, features)}
        return features, feature_dict

    def predict(
        self,
        user: User,
        facts: List[Fact],
        date: datetime = None,
    ) -> np.ndarray:
        if date is None:
            date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
        xs = [self.compute_features(user, fact, date)[0] for fact in facts]
        xs = np.array(xs).astype(np.float32)
        xs = (xs - self.dataset.mean) / self.dataset.std
        ys = []
        batch_size = 128
        for i in range(0, xs.shape[0], batch_size):
            x = xs[i: i + batch_size]
            x = torch.from_numpy(x).to(self.device)
            logits = self.model.forward(x)
            # return the probability of positive (1)
            y = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1]
            ys.append(y)
        return np.concatenate(ys)

    def predict_one(
        self,
        features: RetentionFeatures,
    ) -> float:
        '''recall probability of a single fact'''
        xs = [
            features.user_count_correct,
            features.user_count_wrong,
            features.user_count_total,
            features.user_average_overall_accuracy,
            features.user_average_question_accuracy,
            features.user_previous_result,
            features.user_gap_from_previous,
            features.question_average_overall_accuracy,
            features.question_count_total,
            features.question_count_correct,
            features.question_count_wrong,
            1, # bias
        ]
        xs = np.array(xs).astype(np.float32)
        xs = (xs - self.dataset.mean) / self.dataset.std
        x = torch.from_numpy(xs).to(self.device)
        logits = self.model.forward(x)
        # return the probability of positive (1)
        y = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1][0]
        return y


# create logger with 'retention'
logger = logging.getLogger('retention')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('/Users/shifeng/workspace/fact-repetition/retention.log')
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


@app.get('/api/karl/predict')
def predict_one(features: RetentionFeatures):
    return retention_model.predict_one(features)