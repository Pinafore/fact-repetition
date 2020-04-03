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
    # skill estimate for each topic
    skill: np.ndarray
    # for computing question average accuracy
    # I originally thought we should store (uid, result, date)
    # but that can be quickly inferred from History table
    results: List[bool] = field(default_factory=list)

    @classmethod
    def unpack(cls, r):
        fields = [
            'card_id',
            'text',
            'answer',
            'category',
            'qrep',
            'skill',
            'results'
        ]

        if isinstance(r, str):
            r = json.loads(r)

        if isinstance(r, dict):
            r = [r[f] for f in fields]

        return Card(
            card_id=r[0],
            text=r[1],
            answer=r[2],
            category=r[3],
            qrep=np.array(json.loads(r[4])),
            skill=np.array(json.loads(r[5])),
            results=json.loads(r[6])
        )

    def pack(self):
        x = self
        return (
            x.card_id,
            x.text,
            x.answer,
            x.category,
            json.dumps(x.qrep.tolist()),
            json.dumps(x.skill.tolist()),
            json.dumps(x.results)
        )


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
    n_topics: int = 40
    qrep: float = 1.0
    skill: float = 1.0
    time: float = 1.0
    category: float = 1.0
    leitner: float = 1.0
    sm2: float = 1.0
    decay_qrep: float = 0.9
    max_qreps: int = 10
    cool_down_time: float = 20.
    lda_dir: str = 'checkpoints/gensim_all_40_1585820469.362995'
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

    # for computing user average accuracy
    results: List[bool] = field(default_factory=list)
    # qid -> number of times user and qid correctly
    count_correct_before: Dict[str, int] = field(default_factory=dict)
    # qid -> number of times user and qid incorrectly
    count_wrong_before: Dict[str, int] = field(default_factory=dict)

    def pack(self):
        x = self
        return (
            x.user_id,
            json.dumps([q.tolist() for q in x.qrep]),
            json.dumps([q.tolist() for q in x.skill]),
            x.category,
            json.dumps({k: str(v) for k, v in x.last_study_date.items()}),
            json.dumps(x.leitner_box),
            json.dumps({k: str(v) for k, v in x.leitner_scheduled_date.items()}),
            json.dumps(x.sm2_efactor),
            json.dumps(x.sm2_interval),
            json.dumps(x.sm2_repetition),
            json.dumps({k: str(v) for k, v in x.sm2_scheduled_date.items()}),
            json.dumps(x.results),
            json.dumps(x.count_correct_before),
            json.dumps(x.count_wrong_before),
        )

    @classmethod
    def unpack(cls, r):
        fields = [
            'user_id',
            'qrep',
            'skill',
            'category',
            'last_study_date',
            'leitner_box',
            'leitner_scheduled_date',
            'sm2_efactor',
            'sm2_interval',
            'sm2_repetition',
            'sm2_scheduled_date',
            'results',
            'count_correct_before',
            'count_wrong_before'
        ]

        if isinstance(r, str):
            r = json.loads(r)

        if isinstance(r, dict):
            r = [r[f] for f in fields]

        return User(
            user_id=r[0],
            qrep=[np.array(x) for x in json.loads(r[1])],
            skill=[np.array(x) for x in json.loads(r[2])],
            category=r[3],
            last_study_date={k: parse_date(v) for k, v in json.loads(r[4]).items()},
            leitner_box=json.loads(r[5]),
            leitner_scheduled_date={k: parse_date(v) for k, v in json.loads(r[6]).items()},
            sm2_efactor=json.loads(r[7]),
            sm2_interval=json.loads(r[8]),
            sm2_repetition=json.loads(r[9]),
            sm2_scheduled_date={k: parse_date(v) for k, v in json.loads(r[10]).items()},
            results=json.loads(r[11]),
            count_correct_before=json.loads(r[12]),
            count_wrong_before=json.loads(r[13]),
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
