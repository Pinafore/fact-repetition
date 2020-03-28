#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, List
from dataclasses import dataclass, field


def parse_date(date: str):
    if isinstance(date, datetime):
        return date
    if isinstance(date, str):
        return datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    else:
        raise TypeError("unrecognized type for parse_date")


@dataclass
class Card:
    card_id: str
    text: str
    answer: str
    category: str
    qrep: np.ndarray
    skill: np.ndarray  # skill estimate for each topic


class ScheduleRequest(BaseModel):
    text: str
    date: Optional[str]
    answer: Optional[str]
    category: Optional[str]
    user_id: Optional[str]
    question_id: Optional[str]
    label: Optional[str]
    history_id: Optional[str]


class Params(BaseModel):
    n_topics: int = 10
    qrep: float = 1.0
    skill: float = 1.0
    time: float = 1.0
    category: float = 1.0
    leitner: float = 1.0
    sm2: float = 1.0
    decay_qrep: float = 0.9
    lda_dir: str = 'checkpoints/gensim_quizbowl_10_1585102364.5221019'
    whoosh_index: str = 'whoosh_index'


@dataclass
class History:
    history_id: str
    user_id: str
    card_id: str
    response: str
    judgement: str
    user_snapshot: str
    scheduler_snapshot: str
    card_ids: List[str]
    scheduler_output: str
    date: datetime


@dataclass
class User:
    user_id: str
    # qrep of recently studied cards
    qrep: List[np.ndarray]
    # skill of recently studied cards
    skill: List[np.ndarray]
    category: str
    last_study_date: Dict[str, datetime] = field(default_factory=dict)
    leitner_box: Dict[str, int] = field(default_factory=dict)
    leitner_scheduled_date: Dict[str, datetime] = field(default_factory=dict)
    sm2_efactor: Dict[str, float] = field(default_factory=dict)
    sm2_interval: Dict[str, float] = field(default_factory=dict)
    sm2_repetition: Dict[str, int] = field(default_factory=dict)
    sm2_scheduled_date: Dict[str, datetime] = field(default_factory=dict)

    def to_snapshot(self):
        x = self.__dict__.copy()
        # user_id: str
        x['qrep'] = [q.tolist() for q in x['qrep']]
        x['skill'] = [q.tolist() for q in x['skill']]
        # category: str
        x['last_study_date'] = {k: str(v) for k, v in x['last_study_date'].items()}
        # leitner_box: Dict[str, int]
        x['leitner_scheduled_date'] = {k: str(v) for k, v in x['leitner_scheduled_date'].items()}
        # sm2_efactor: Dict[str, float]
        # sm2_interval: Dict[str, float]
        # sm2_repetition: Dict[str, int]
        x['sm2_scheduled_date'] = {k: str(v) for k, v in x['sm2_scheduled_date'].items()}
        return json.dumps(x)

    @classmethod
    def from_snapshot(cls, s):
        x = json.loads(s)
        return User(
            user_id=x['user_id'],
            qrep=[np.array(q) for q in x['qrep']],
            skill=[np.array(q) for q in x['skill']],
            category=x['category'],
            last_study_date={k: parse_date(v) for k, v in x['last_study_date'].items()},
            leitner_box=x['leitner_box'],
            leitner_scheduled_date={k: parse_date(v) for k, v in x['leitner_scheduled_date'].items()},
            sm2_efactor=x['sm2_efactor'],
            sm2_interval=x['sm2_interval'],
            sm2_repetition=x['sm2_repetition'],
            sm2_scheduled_date={k: parse_date(v) for k, v in x['sm2_scheduled_date'].items()}
        )


# class Flashcard(BaseModel):
#     text: str
#     user_id: Optional[str]
#     question_id: Optional[str]
#     user_accuracy: Optional[float]
#     user_buzzratio: Optional[float]
#     user_count: Optional[float]
#     question_accuracy: Optional[float]
#     question_buzzratio: Optional[float]
#     question_count: Optional[float]
#     times_seen: Optional[float]
#     times_correct: Optional[float]
#     times_wrong: Optional[float]
#     label: Optional[str]
#     answer: Optional[str]
#     category: Optional[str]
