#!/usr/bin/env python
# coding: utf-8

import torch
import logging
import numpy as np
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import torch.nn.functional as F

from karl.retention.data import RetentionDataset
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

    def predict_one(
        self,
        feature_vector: RetentionFeatures,
    ) -> float:
        '''recall probability of a single fact'''
        xs = [[
            feature_vector.user_count_correct,
            feature_vector.user_count_wrong,
            feature_vector.user_count_total,
            feature_vector.user_average_overall_accuracy,
            feature_vector.user_average_question_accuracy,
            feature_vector.user_previous_result,
            feature_vector.user_gap_from_previous,
            feature_vector.question_average_overall_accuracy,
            feature_vector.question_count_total,
            feature_vector.question_count_correct,
            feature_vector.question_count_wrong,
            1,  # bias
        ]]
        xs = np.array(xs).astype(np.float32)
        xs = (xs - self.dataset.mean) / self.dataset.std
        x = torch.from_numpy(xs).to(self.device)
        logits = self.model.forward(x)
        # return the probability of positive (1)
        y = F.softmax(logits, dim=1).detach().cpu().numpy()[:, 1][0].item()
        return y

    def predict(
        self,
        feature_vectors: List[RetentionFeatures],
    ) -> List[float]:
        xs = [[
            feature_vector.user_count_correct,
            feature_vector.user_count_wrong,
            feature_vector.user_count_total,
            feature_vector.user_average_overall_accuracy,
            feature_vector.user_average_question_accuracy,
            feature_vector.user_previous_result,
            feature_vector.user_gap_from_previous,
            feature_vector.question_average_overall_accuracy,
            feature_vector.question_count_total,
            feature_vector.question_count_correct,
            feature_vector.question_count_wrong,
            1,  # bias
        ] for feature_vector in feature_vectors]
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
        return np.concatenate(ys).tolist()


# create logger with 'retention'
logger = logging.getLogger('retention')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('/fs/clip-quiz/shifeng/karl-dev/retention.log')
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
def predict_one(feature_vector: RetentionFeatures):
    return retention_model.predict_one(feature_vector)

@app.get('/api/karl/predict')
def predict(feature_vectors: List[RetentionFeatures]):
    return retention_model.predict(feature_vectors)
