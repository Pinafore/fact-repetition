import json
import numpy as np
from datetime import datetime
from typing import NamedTuple, Dict, List


class Card(NamedTuple):
    card_id: str
    text: str
    answer: str
    qrep: np.ndarray
    skill: float
    category: str
    last_update: datetime


class History(NamedTuple):
    history_id: str
    user_id: str
    card_id: str
    response: str
    judgement: str
    user_snapshot: str
    scheduler_snapshot: str
    cards: List[str]
    scheduler_output: str
    timestamp: datetime


class User(NamedTuple):
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
        x = self._asdict()
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
