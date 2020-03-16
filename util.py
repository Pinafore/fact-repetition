import json
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, List
from dataclasses import dataclass, field


@dataclass
class Card:
    card_id: str
    text: str
    answer: str
    qrep: np.ndarray
    skill: np.ndarray
    category: str
    date: datetime


class Flashcard(BaseModel):
    text: str
    answer: str
    category: str
    user_id: Optional[str]
    question_id: Optional[str]
    label: Optional[str]
    history_id: Optional[str]


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


class Params(BaseModel):
    n_components: int = 20
    qrep: float = 0.1
    skill: float = 0.7
    category: float = 0.3
    leitner: float = 1.0
    sm2: float = 1.0
    step_correct: float = 0.5
    step_wrong: float = 0.05
    step_qrep: float = 0.3
    vectorizer: str = 'checkpoints/tf_vectorizer.pkl'
    lda: str = 'checkpoints/lda.pkl'
    whoosh_index: str = 'whoosh_index'

    class Config:
        arbitrary_types_allowed = True


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
    qrep: np.ndarray
    skill: np.ndarray
    category: str
    repetition: Dict[str, int] = field(default_factory=dict)
    last_study_time: Dict[str, datetime] = field(default_factory=dict)
    leitner_box: Dict[str, int] = field(default_factory=dict)
    leitner_scheduled_time: Dict[str, datetime] = field(default_factory=dict)
    sm2_efactor: Dict[str, float] = field(default_factory=dict)
    sm2_interval: Dict[str, float] = field(default_factory=dict)
    sm2_scheduled_time: Dict[str, datetime] = field(default_factory=dict)
    date: datetime = field(default_factory=datetime.now)

    def to_snapshot(self):
        x = self.__dict__.copy()
        # user_id: str
        x['qrep'] = x['qrep'].tolist()
        x['skill'] = x['skill'].tolist()
        # category: str
        # repetition: Dict[str, int]
        x['last_study_time'] = {k: str(v) for k, v in x['last_study_time'].items()}
        # leitner_box: Dict[str, int]
        x['leitner_scheduled_time'] = {k: str(v) for k, v in x['leitner_scheduled_time'].items()}
        # sm2_efactor: Dict[str, float]
        # sm2_interval: Dict[str, float]
        x['sm2_scheduled_time'] = {k: str(v) for k, v in x['sm2_scheduled_time'].items()}
        x['date'] = str(x['date'])
        return json.dumps(x)

    @classmethod
    def from_snapshot(cls, s):
        x = json.loads(s)
        return User(
            user_id=x['user_id'],
            qrep=np.array(x['qrep']),
            skill=np.array(x['skill']),
            category=x['category'],
            repetition=x['repetition'],
            last_study_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                             for k, v in x['last_study_time'].items()},
            leitner_box=x['leitner_box'],
            leitner_scheduled_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                                    for k, v in x['leitner_scheduled_time'].items()},
            sm2_efactor=x['sm2_efactor'],
            sm2_interval=x['sm2_interval'],
            sm2_scheduled_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                                for k, v in x['sm2_scheduled_time'].items()},
            date=datetime.strptime(x['date'], "%Y-%m-%d %H:%M:%S.%f")
        )
