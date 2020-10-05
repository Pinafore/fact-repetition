# %%
#!/usr/bin/env python
# coding: utf-8

import json
import atexit
import socket
import logging
from fastapi import FastAPI
from typing import Optional, List
from pydantic import BaseModel
from datetime import datetime
from dateutil.parser import parse as parse_date
from cachetools import cached, TTLCache
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import SQLAlchemyError

from karl.new_util import ScheduleRequest, Params
from karl.scheduler import MovingAvgScheduler


app = FastAPI()
scheduler = MovingAvgScheduler(preemptive=False)

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

    return {env: sessionmaker(bind=engine, autoflush=False)() for env, engine in engines.items()}


sessions = get_sessions()

# create logger with 'scheduler'
logger = logging.getLogger('scheduler')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('/fs/www-users/shifeng/scheduler.log')
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


@app.post('/api/karl/schedule')
def schedule(requests: List[ScheduleRequest]):
    if len(requests) == 0:
        return {
            'order': [],
            'rationale': '<p>no fact received</p>',
            'facts_info': '',
        }

    logger.info(f'/karl/schedule with {len(requests)} facts and env={requests[0].env}')

    # NOTE assuming single user single date
    date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
    if requests[0].date is not None:
        date = parse_date(requests[0].date)

    env = 'dev' if requests[0].env == 'dev' else 'prod'

    try:
        results = scheduler.schedule(sessions[env], requests, date, plot=False)
        return {
            'order': results['order'],
            'rationale': results['rationale'],
            'facts_info': results['facts_info'],
            # 'profile': results['profile'],
        }
    except SQLAlchemyError as e:
        print(repr(e))
        sessions[env].rollback()


@app.post('/api/karl/update')
def update(requests: List[ScheduleRequest]):
    logger.info(f'/karl/update with {len(requests)} facts and env={requests[0].env}')

    # NOTE assuming single user single date
    date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
    if requests[0].date is not None:
        date = parse_date(requests[0].date)

    env = 'dev' if requests[0].env == 'dev' else 'prod'

    try:
        scheduler.update(sessions[env], requests, date)
        sessions[env].commit()
    except SQLAlchemyError as e:
        print(repr(e))
        sessions[env].rollback()


@app.put('/api/karl/set_params')
def set_params(user_id: str, env: str, params: Params):
    env = 'dev' if env == 'dev' else 'prod'
    session = sessions[env]
    try:
        scheduler.set_user_params(session, user_id, params)
        session.commit()
    except SQLAlchemyError as e:
        print(repr(e))
        session.rollback()


@app.get('/api/karl/set_repetition_model')
def set_repetition_model(user_id: str, env: str, repetition_model: str):
    env = 'dev' if env == 'dev' else 'prod'
    session = sessions[env]
    if repetition_model == 'sm2':
        params = Params(
            repetition_model='sm2',
            qrep=0,
            skill=0,
            recall=0,
            category=0,
            answer=0,
            leitner=0,
            sm2=1,
            cool_down=0,
        )
        scheduler.set_user_params(session, user_id, params)
        session.commit()
    elif repetition_model == 'leitner':
        params = Params(
            repetition_model='leitner',
            qrep=0,
            skill=0,
            recall=0,
            category=0,
            answer=0,
            leitner=1,
            sm2=0,
            cool_down=0,
        )
        scheduler.set_user_params(session, user_id, params)
        session.commit()
    elif repetition_model.startswith('karl'):
        recall_target = int(repetition_model[4:])
        params = Params(
            repetition_model=f'karl{recall_target}',
            qrep=1,
            skill=0,
            recall=1,
            category=1,
            answer=1,
            leitner=1,
            sm2=0,
            recall_target=float(recall_target) / 100,
        )
        scheduler.set_user_params(session, user_id, params)
        session.commit()
    else:
        pass


@app.post('/api/karl/get_fact')
def get_fact(request: ScheduleRequest):
    env = 'dev' if request.env == 'dev' else 'prod'
    fact = scheduler.get_fact(sessions[env], request)
    return json.dumps({
        k: v for k, v in fact.__dict__.items() if k != '_sa_instance_state'
    })


@app.get('/api/karl/reset_user')
def reset_user(user_id: str = None, env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    try:
        scheduler.reset_user(sessions[env], user_id=user_id)
        sessions[env].commit()
    except SQLAlchemyError as e:
        print(repr(e))
        sessions[env].rollback()


@app.get('/api/karl/reset_fact')
def reset_fact(fact_id: str = None, env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    try:
        scheduler.reset_fact(sessions[env], fact_id=fact_id)
        sessions[env].commit()
    except SQLAlchemyError as e:
        print(repr(e))
        sessions[env].rollback()


@app.get('/api/karl/status')
def status():
    return True


@app.get('/api/karl/get_user')
def get_user(user_id: str, env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    user = scheduler.get_user(sessions[env], user_id)
    return json.dumps({
        k: v for k, v in user.__dict__.items() if k != '_sa_instance_state'
    })


@app.get('/api/karl/get_all_users')
def get_all_users(env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    users = scheduler.get_all_users(sessions[env])
    return [json.dumps({
        k: v for k, v in user.__dict__.items() if k != '_sa_instance_state'
    }) for user in users]


@app.get('/api/karl/get_user_history')
def get_user_history(user_id: str, env: str = None, deck_id: str = None,
                     date_start: str = None, date_end: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    return scheduler.get_records(sessions[env], user_id, deck_id, date_start, date_end)


@app.get('/api/karl/get_user_stats')
# @cached(cache=TTLCache(maxsize=1024, ttl=600))
def get_user_stats(user_id: str, env: str = None, deck_id: str = None,
                   date_start: str = None, date_end: str = None):
    '''
    Return in a dictionary the following user stats within given date range.

    new_facts: int
    reviewed_facts: int
    total_seen: int
    total_seconds: int
    known_rate: float
    new_known_rate: float
    review_known_rate: float
    '''
    env = 'dev' if env == 'dev' else 'prod'
    return scheduler.get_user_stats(sessions[env], user_id, deck_id, date_start, date_end)

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


@app.get('/api/karl/leaderboard')
# @cached(cache=TTLCache(maxsize=1024, ttl=1800))
def leaderboard(
        user_id: str = None,
        env: str = None,
        skip: int = 0,
        limit: int = 10,
        rank_type: str = 'total_seen',
        min_studied: int = 0,
        deck_id: str = None,
        date_start: str = None,
        date_end: str = None,
):
    '''
    return [(user_id: str, rank_type: 'total_seen', value: 'value')]
    that ranks [skip: skip + limit)
    '''
    env = 'dev' if env == 'dev' else 'prod'

    stats = {}
    for user in scheduler.get_all_users(sessions[env]):
        if not user.user_id.isdigit():
            continue

        stats[user.user_id] = get_user_stats(
            user_id=user.user_id,
            env=env,
            deck_id=deck_id,
            date_start=date_start,
            date_end=date_end
        )

    # profile_keys = [
    #     'profile_get_history_records',
    #     'profile_compute_stats',
    #     'profile_n_history_records',
    # ]
    # profile = {
    #     key: np.sum([v[key] for v in stats.values()])
    #     for key in profile_keys
    # }

    # from high value to low
    stats = sorted(stats.items(), key=lambda x: x[1][rank_type])[::-1]
    stats = [(k, v) for k, v in stats if v['total_seen'] >= min_studied]

    rankings = []
    user_place = None
    for i, (k, v) in enumerate(stats):
        if user_id == k:
            user_place = i
        rankings.append(Ranking(user_id=k, rank=i + 1, value=v[rank_type]))

    leaderboard = Leaderboard(
        leaderboard=rankings[skip: skip + limit],
        total=len(rankings),
        rank_type=rank_type,
        user_place=user_place,
        user_id=user_id,
        skip=skip,
        limit=limit,
    )

    # pprint(profile)
    return leaderboard


@atexit.register
def finalize_db():
    pass
    # for scheduler in schedulers.values():
    #     scheduler.db.commit()
    #     scheduler.db.close()