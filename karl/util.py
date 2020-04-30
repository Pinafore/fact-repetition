#!/usr/bin/env python
# coding: utf-8

import json
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, List
from dataclasses import dataclass, field

from plotnine import theme, theme_light, \
    element_text, element_blank, element_rect, element_line


def parse_date(date: str):
    if isinstance(date, datetime):
        return date
    if isinstance(date, str):
        return datetime.strptime(date, "%Y-%m-%d %H:%M:%S.%f")
    else:
        raise TypeError("unrecognized type for parse_date")


class Params(BaseModel):
    qrep: float = 1                     # cosine distance between qreps
    skill: float = 0                    # fact difficulty vs user skill level
    recall: float = 1                   # recall probability
    category: float = 1                 # change in category from prev
    leitner: float = 0                  # hours till leitner scheduled date
    sm2: float = 1                      # hours till sm2 scheduled date
    decay_qrep: float = 0.9             # discount factor
    decay_skill: float = 0.9            # discount factor
    cool_down: float = 1                # weight for cool down
    cool_down_time_correct: float = 20  # minutes to cool down
    cool_down_time_wrong: float = 4     # minutes to cool down
    max_queue: int = 10                 # num of qrep/skill vectors to average over


leitner_params = Params(
    qrep=0,
    skill=0,
    recall=0,
    category=0,
    leitner=1,
    sm2=0,
    cool_down=0,
)

sm2_params = Params(
    qrep=0,
    skill=0,
    recall=0,
    category=0,
    leitner=0,
    sm2=1,
    cool_down=0,
)


@dataclass
class Fact:
    fact_id: str
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
            'fact_id',
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

        return Fact(
            fact_id=r[0],
            text=r[1],
            answer=r[2],
            category=r[3],
            qrep=np.array(json.loads(r[4])),
            skill=np.array(json.loads(r[5])),
            results=json.loads(r[6])
        )

    def pack(self):
        x = self
        return [
            x.fact_id,
            x.text,
            x.answer,
            x.category,
            json.dumps(x.qrep.tolist()),
            json.dumps(x.skill.tolist()),
            json.dumps(x.results)
        ]


class ScheduleRequest(BaseModel):
    text: str
    date: Optional[str]
    answer: Optional[str]
    category: Optional[str]
    user_id: Optional[str]
    fact_id: Optional[str]
    label: Optional[str]
    history_id: Optional[str]
    repetition_model: Optional[str]


@dataclass
class History:
    history_id: str
    user_id: str
    fact_id: str
    response: str
    judgement: str
    user_snapshot: str
    scheduler_snapshot: str
    fact_ids: List[str]
    scheduler_output: str
    date: datetime


@dataclass
class User:
    user_id: str
    # qrep of recently studied Facts
    qrep: List[np.ndarray]
    # skill of recently studied Facts
    skill: List[np.ndarray]
    category: str
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

    def pack(self):
        x = self
        return [
            x.user_id,
            json.dumps([q.tolist() for q in x.qrep]),
            json.dumps([q.tolist() for q in x.skill]),
            x.category,
            json.dumps({k: (str(v), r) for k, (v, r) in x.previous_study.items()}),
            json.dumps(x.leitner_box),
            json.dumps({k: str(v) for k, v in x.leitner_scheduled_date.items()}),
            json.dumps(x.sm2_efactor),
            json.dumps(x.sm2_interval),
            json.dumps(x.sm2_repetition),
            json.dumps({k: str(v) for k, v in x.sm2_scheduled_date.items()}),
            json.dumps(x.results),
            json.dumps(x.count_correct_before),
            json.dumps(x.count_wrong_before),
            json.dumps(x.params.__dict__),
        ]

    @classmethod
    def unpack(cls, r):
        fields = [
            'user_id',
            'qrep',
            'skill',
            'category',
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
            previous_study={k: (parse_date(v), r) for k, (v, r) in json.loads(r[4]).items()},
            leitner_box=json.loads(r[5]),
            leitner_scheduled_date={k: parse_date(v) for k, v in json.loads(r[6]).items()},
            sm2_efactor=json.loads(r[7]),
            sm2_interval=json.loads(r[8]),
            sm2_repetition=json.loads(r[9]),
            sm2_scheduled_date={k: parse_date(v) for k, v in json.loads(r[10]).items()},
            results=json.loads(r[11]),
            count_correct_before=json.loads(r[12]),
            count_wrong_before=json.loads(r[13]),
            params=Params(**json.loads(r[14])),
        )


class theme_fs(theme_light):
    """
    A theme similar to :class:`theme_linedraw` but with light grey
    lines and axes to direct more attention towards the data.
    Parameters
    ----------
    base_size : int, optional
        Base font size. All text sizes are a scaled versions of
        the base font size. Default is 11.
    base_family : str, optional
        Base font family.
    """

    def __init__(self, base_size=11, base_family='DejaVu Sans'):
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
