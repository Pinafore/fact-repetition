#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from dateutil.parser import parse as parse_date

from plotnine import theme, theme_light, \
    element_text, element_blank, element_rect, element_line


class SetParams(BaseModel):
    user_id: str = None                 # make it easier to set params for user
    env: str = None                     # make it easier to set params for user
    qrep: float = 1                     # cosine distance between qreps
    skill: float = 0                    # fact difficulty vs user skill level
    recall: float = 1                   # recall probability
    recall_target: float = 1            # recall target probability
    category: float = 1                 # change in category from prev
    answer: float = 1                   # reptition of the same category
    leitner: float = 0                  # hours till leitner scheduled date
    sm2: float = 1                      # hours till sm2 scheduled date
    decay_qrep: float = 0.9             # discount factor
    decay_skill: float = 0.9            # discount factor
    cool_down: float = 1                # weight for cool down
    cool_down_time_correct: float = 20  # minutes to cool down
    cool_down_time_wrong: float = 4     # minutes to cool down
    max_recent_facts: int = 10          # num of recent facts to keep record of


@dataclass
class Params:
    qrep: float = 1                     # cosine distance between qreps
    skill: float = 0                    # fact difficulty vs user skill level
    recall: float = 1                   # recall probability
    recall_target: float = 1            # target of recall probability
    category: float = 1                 # change in category from prev
    answer: float = 1                   # reptition of the same category
    leitner: float = 0                  # hours till leitner scheduled date
    sm2: float = 1                      # hours till sm2 scheduled date
    decay_qrep: float = 0.9             # discount factor
    decay_skill: float = 0.9            # discount factor
    cool_down: float = 1                # weight for cool down
    cool_down_time_correct: float = 20  # minutes to cool down
    cool_down_time_wrong: float = 4     # minutes to cool down
    max_recent_facts: int = 10          # num of recent facts to keep record of


@dataclass
class Fact:
    fact_id: str
    text: str
    answer: str
    category: str
    deck_name: Optional[str]
    deck_id: Optional[str]
    qrep: Optional[np.ndarray]
    # skill estimate for each topic
    skill: Optional[np.ndarray]
    # for computing question average accuracy
    # I originally thought we should store (uid, result, date)
    # but that can be quickly inferred from History table
    results: List[bool] = field(default_factory=list)

    @classmethod
    def unpack(cls, r):
        fields = [
            'fact_id',
            'text',
            'answer',
            'category',
            'deck_name',
            'deck_id',
            'qrep',
            'skill',
            'results'
        ]

        if isinstance(r, str):
            r = json.loads(r)

        if isinstance(r, dict):
            r = [r[f] for f in fields]

        return Fact(
            fact_id=r[0],
            text=r[1],
            answer=r[2],
            category=r[3],
            deck_name=r[4],
            deck_id=r[5],
            qrep=np.array(json.loads(r[6])),
            skill=np.array(json.loads(r[7])),
            results=json.loads(r[8])
        )

    def pack(self):
        x = self
        return [
            x.fact_id,
            x.text,
            x.answer,
            x.category,
            x.deck_name,
            x.deck_id,
            json.dumps(x.qrep.tolist()),
            json.dumps(x.skill.tolist()),
            json.dumps(x.results)
        ]


@dataclass
class UserStats:
    new_facts: int = 0
    reviewed_facts: int = 0
    total_seen: int = 0
    total_seconds: int = 0
    last_week_seen: int = 0
    last_week_new_facts: int = 0
    new_known_rate: float = 0
    review_known_rate: float = 0
    # correct / wrong, new card / review, datetime string
    results: List[tuple] = field(default_factory=list)


@dataclass
class User:
    # TODO omg so much bloat
    user_id: str
    recent_facts: List[Fact] = field(default_factory=list)
    previous_study: dict = field(default_factory=dict)

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

    params: Params = field(default_factory=Params)
    # user_stats: UserStats = field(default_factory=UserStats)

    def pack(self):
        x = self
        return [
            x.user_id,
            json.dumps([f.pack() for f in x.recent_facts]),
            json.dumps({k: (v.strftime('%Y-%m-%dT%H:%M:%S%z'), r) for k, (v, r) in x.previous_study.items()}),
            json.dumps(x.leitner_box),
            json.dumps({k: v.strftime('%Y-%m-%dT%H:%M:%S%z') for k, v in x.leitner_scheduled_date.items()}),
            json.dumps(x.sm2_efactor),
            json.dumps(x.sm2_interval),
            json.dumps(x.sm2_repetition),
            json.dumps({k: v.strftime('%Y-%m-%dT%H:%M:%S%z') for k, v in x.sm2_scheduled_date.items()}),
            json.dumps(x.results),
            json.dumps(x.count_correct_before),
            json.dumps(x.count_wrong_before),
            json.dumps(x.params.__dict__),
            # json.dumps(x.user_stats.__dict__),
        ]

    @classmethod
    def unpack(cls, r):
        fields = [
            'user_id',
            'recent_facts',
            'previous_study',
            'leitner_box',
            'leitner_scheduled_date',
            'sm2_efactor',
            'sm2_interval',
            'sm2_repetition',
            'sm2_scheduled_date',
            'results',
            'count_correct_before',
            'count_wrong_before',
            'params',
            # 'user_stats',
        ]

        if isinstance(r, str):
            r = json.loads(r)

        if isinstance(r, dict):
            r = [r[f] for f in fields]

        return User(
            user_id=r[0],
            recent_facts=[Fact.unpack(f) for f in json.loads(r[1])],
            previous_study={
                k: (parse_date(v), r)
                for k, (v, r) in json.loads(r[2]).items()
            },
            leitner_box=json.loads(r[3]),
            leitner_scheduled_date={k: parse_date(v) for k, v in json.loads(r[4]).items()},
            sm2_efactor=json.loads(r[5]),
            sm2_interval=json.loads(r[6]),
            sm2_repetition=json.loads(r[7]),
            sm2_scheduled_date={k: parse_date(v) for k, v in json.loads(r[8]).items()},
            results=json.loads(r[9]),
            count_correct_before=json.loads(r[10]),
            count_wrong_before=json.loads(r[11]),
            params=Params(**json.loads(r[12])),
            # user_stats=UserStats(**json.loads(r[13])),
        )


class ScheduleRequest(BaseModel):
    text: str
    date: Optional[str]
    answer: Optional[str]
    category: Optional[str]
    user_id: Optional[str]
    fact_id: Optional[str]
    label: Optional[bool]
    history_id: Optional[str]
    repetition_model: Optional[str]
    deck_name: Optional[str]
    deck_id: Optional[str]
    env: Optional[str]
    elapsed_seconds_text: Optional[int]
    elapsed_seconds_answer: Optional[int]
    elapsed_milliseconds_text: Optional[int]
    elapsed_milliseconds_answer: Optional[int]


@dataclass
class History:
    history_id: str
    debug_id: str
    user_id: str
    fact_id: str
    deck_id: str
    response: str
    judgement: str
    user_snapshot: str
    scheduler_snapshot: str
    fact_ids: List[str]
    scheduler_output: str
    elapsed_seconds_text: int = 0
    elapsed_seconds_answer: int = 0
    elapsed_milliseconds_text: int = 0
    elapsed_milliseconds_answer: int = 0
    is_new_fact: int = 0
    date: datetime = None


class theme_fs(theme_light):
    """
    A theme similar to :class:`theme_linedraw` but with light grey
    lines and axes to direct more attention towards the data.
    Parameters
    """

    def __init__(self, base_size=11, base_family='DejaVu Sans'):
        """
        :param base_size: All text sizes are a scaled versions of the base font size.
        :param base_family: Base font family.
        """
        theme_light.__init__(self, base_size, base_family)
        self.add_theme(theme(
            axis_ticks=element_line(color='#DDDDDD', size=0.5),
            panel_border=element_rect(fill='None', color='#838383',
                                      size=1),
            strip_background=element_rect(
                fill='#DDDDDD', color='#838383', size=1),
            strip_text_x=element_text(color='black'),
            strip_text_y=element_text(color='black', angle=-90),
            legend_key=element_blank(),
        ), inplace=True)
