import json
import numpy as np
from datetime import datetime
from pydantic import BaseModel
from typing import Optional, Dict, List
from dataclasses import dataclass
try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7


'''
BaseModels are used for fastapi
NamedTuples and TypedDicts are used for DB
'''


class Flashcard(BaseModel):
    text: str
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


@dataclass
class Params:
    n_components: int = 20
    qrep: float = 0.1
    prob: float = 0.7
    category: float = 0.3
    leitner: float = 1.0
    sm2: float = 1.0
    step_correct: float = 0.5
    step_wrong: float = 0.05
    step_qrep: float = 0.3
    vectorizer: str = 'checkpoints/tf_vectorizer.pkl'
    lda: str = 'checkpoints/lda.pkl'
    whoosh_index: str = 'whoosh_index'


class Hyperparams(BaseModel):
    # TODO merge this with Params
    n_components: Optional[int]
    qrep: Optional[float]
    prob: Optional[float]
    category: Optional[float]
    leitner: Optional[float]
    sm2: Optional[float]
    step_correct: Optional[float]
    step_wrong: Optional[float]
    step_qrep: Optional[float]
    vectorizer: Optional[str]
    lda: Optional[str]
    whoosh_index: Optional[str]


@dataclass
class Card:
    card_id: str
    text: str
    answer: str
    qrep: np.ndarray
    skill: float
    category: str
    last_update: datetime


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
    timestamp: datetime


@dataclass
class User:
    user_id: str
    qrep: np.ndarray
    skill: np.ndarray
    repetition: Dict[str, int]
    last_study_time: Dict[str, datetime]
    scheduled_time: Dict[str, datetime]
    sm2_efactor: Dict[str, float]
    sm2_interval: Dict[str, float]
    leitner_box: Dict[str, int]
    last_update: datetime

    def to_snapshot(self):
        x = self.__dict__.copy()
        x['qrep'] = x['qrep'].tolist()
        x['skill'] = x['skill'].tolist()
        # repetition: Dict[str, int]
        x['last_study_time'] = {k: str(v) for k, v in x['last_study_time'].items()}
        x['scheduled_time'] = {k: str(v) for k, v in x['scheduled_time'].items()}
        # sm2_efactor: Dict[str, float]
        # sm2_interval: Dict[str, float]
        # leitner_box: Dict[str, int]
        x['last_update'] = str(x['last_update'])
        return json.dumps(x)

    @classmethod
    def from_snapshot(cls, s):
        x = json.loads(s)
        return User(
            user_id=x['user_id'],
            qrep=np.array(x['qrep']),
            skill=np.array(x['skill']),
            repetition=x['repetition'],
            last_study_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                             for k, v in x['last_study_time'].items()},
            scheduled_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                            for k, v in x['scheduled_time'].items()},
            sm2_efactor=x['sm2_efactor'],
            sm2_interval=x['sm2_interval'],
            leitner_box=x['leitner_box'],
            last_update=datetime.strptime(x['last_update'], "%Y-%m-%d %H:%M:%S.%f")
        )
