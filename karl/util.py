import json
import socket
import datetime
from typing import Optional, List
from pydantic import BaseModel
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


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


class Params(BaseModel):
    repetition_model: str = 'karl100'   # name of the repetition model
    qrep: float = 1                     # cosine distance between qreps
    skill: float = 0                    # fact difficulty vs user skill level
    recall: float = 1                   # recall probability
    recall_target: float = 1            # target of recall probability
    category: float = 1                 # change in category from prev
    answer: float = 1                   # reptition of the same answer
    leitner: float = 0                  # hours till leitner scheduled date
    sm2: float = 1                      # hours till sm2 scheduled date
    decay_qrep: float = 0.9             # discount factor
    decay_skill: float = 0.9            # discount factor
    cool_down: float = 1                # weight for cool down
    cool_down_time_correct: float = 20  # minutes to cool down
    cool_down_time_wrong: float = 4     # minutes to cool down
    max_recent_facts: int = 10          # num of recent facts to keep record of


class IntOrFloat:

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, float) or isinstance(v, int):
            return v
        raise TypeError('int or float required')


class Ranking(BaseModel):
    user_id: int
    rank: int
    # value: Union[int, float]  # don't use this
    value: IntOrFloat


class Leaderboard(BaseModel):
    leaderboard: List[Ranking]
    total: int
    rank_type: str
    user_place: Optional[int] = None
    user_id: Optional[str] = None
    skip: Optional[int] = 0
    limit: Optional[int] = None


class UserStatSchema(BaseModel):
    user_id: str
    deck_id: str
    date_start: str
    date_end: str
    new_facts: Optional[int] = 0
    reviewed_facts: Optional[int] = 0
    new_correct: Optional[int] = 0
    reviewed_correct: Optional[int] = 0
    total_seen: Optional[int] = 0
    total_milliseconds: Optional[int] = 0
    total_seconds: Optional[int] = 0
    total_minutes: Optional[int] = 0
    elapsed_milliseconds_text: Optional[int] = 0
    elapsed_milliseconds_answer: Optional[int] = 0
    elapsed_seconds_text: Optional[int] = 0
    elapsed_seconds_answer: Optional[int] = 0
    elapsed_minutes_text: Optional[int] = 0
    elapsed_minutes_answer: Optional[int] = 0
    n_days_studied: Optional[int] = 0
    known_rate: Optional[float] = 0
    new_known_rate: Optional[float] = 0
    review_known_rate: Optional[float] = 0


def get_sessions():
    hostname = socket.gethostname()
    if hostname.startswith('newspeak'):
        db_host = '/fs/clip-quiz/shifeng/postgres/run'
    elif hostname.startswith('lapine'):
        db_host = '/fs/clip-scratch/shifeng/postgres/run'
    else:
        print('unrecognized hostname')
        exit()
    engines = {
        'prod': create_engine(f'postgresql+psycopg2://shifeng@localhost:5433/karl-prod?host={db_host}'),
        'dev': create_engine(f'postgresql+psycopg2://shifeng@localhost:5433/karl-dev?host={db_host}'),
    }
    return {
        env: sessionmaker(bind=engine, autoflush=False)()
        for env, engine in engines.items()
    }