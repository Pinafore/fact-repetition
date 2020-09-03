#!/usr/bin/env python
# coding: utf-8

import json
import atexit
import logging
import numpy as np
from pprint import pprint
from fastapi import FastAPI
from typing import Optional, List, Union
from pydantic import BaseModel
from datetime import datetime
from dateutil.parser import parse as parse_date
from cachetools import cached, TTLCache

from karl.util import ScheduleRequest, SetParams, Params
from karl.new_util import User, Record
from karl.scheduler import MovingAvgScheduler


app = FastAPI()
schedulers = {
    'dev': MovingAvgScheduler(db_filename='db.sqlite.dev'),
    'prod': MovingAvgScheduler(db_filename='db.sqlite.prod'),
}


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
    scheduler = schedulers[env]

    results = scheduler.schedule(requests, date, plot=False)
    return {
        'order': results['order'],
        'rationale': results['rationale'],
        'facts_info': results['facts_info'],
        'profile': results['profile'],
    }


@app.post('/api/karl/update')
def update(requests: List[ScheduleRequest]):
    logger.info(f'/karl/update with {len(requests)} facts and env={requests[0].env}')

    # NOTE assuming single user single date
    date = parse_date(datetime.now().strftime('%Y-%m-%dT%H:%M:%S%z'))
    if requests[0].date is not None:
        date = parse_date(requests[0].date)

    env = 'dev' if requests[0].env == 'dev' else 'prod'
    scheduler = schedulers[env]

    return scheduler.update(requests, date)


@app.post('/api/karl/set_params')
def set_params(params: SetParams):
    params = params.dict()
    user_id = params.pop('user_id')
    env = params.pop('env')
    env = 'dev' if env == 'dev' else 'prod'
    params = Params(**params)
    schedulers[env].set_user_params(user_id, params)


@app.post('/api/karl/get_fact')
def get_fact(request: ScheduleRequest):
    env = 'dev' if request.env == 'dev' else 'prod'
    return schedulers[env].get_fact(request).pack()


@app.get('/api/karl/reset_user')
def reset_user(user_id: str = None, env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    schedulers[env].reset_user(user_id=user_id)


@app.get('/api/karl/reset_fact')
def reset_fact(fact_id: str = None, env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    schedulers[env].reset_fact(fact_id=fact_id)


@app.get('/api/karl/status')
def status():
    return True


@app.get('/api/karl/get_user')
def get_user(user_id: str, env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    return schedulers[env].get_user(user_id).pack()


@app.get('/api/karl/get_all_users')
def get_all_users(env: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    users = schedulers[env].db.query(User).all()
    # TODO maybe slow
    return [json.dumps({
        k: v for k, v in user.__dict__.items() if k != '_sa_instance_state'
    }) for user in users]

@app.get('/api/karl/get_user_history')
def get_user_history(user_id: str, env: str = None, deck_id: str = None,
                     date_start: str = None, date_end: str = None):
    env = 'dev' if env == 'dev' else 'prod'
    scheduler = schedulers[env]

    if date_start is None:
        date_start = '2008-06-11 08:00:00'
    if date_end is None:
        date_end = '2038-06-11 08:00:00'

    date_start = parse_date(date_start)
    date_end = parse_date(date_end)

    records = scheduler.db.query(Record).\
        filter(Record.user_id == user_id).\
        filter(Record.date >= date_start, Record.date <= date_end)
    if deck_id is not None:
        records = records.filter(Record.deck_id == deck_id)
    return records.all()

@app.get('/api/karl/get_user_stats')
@cached(cache=TTLCache(maxsize=1024, ttl=1800))
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

    t0 = datetime.now()
    history_records = get_user_history(user_id, env, deck_id, date_start, date_end)
    t1 = datetime.now()

    new_facts = 0
    reviewed_facts = 0
    total_seen = 0
    new_known_rate = 0
    review_known_rate = 0
    elapsed_milliseconds_text = 0
    elapsed_milliseconds_answer = 0

    new_known, review_known, overall_known = [], [], []
    for h in history_records:
        if h.is_new_fact:
            new_facts += 1
            new_known.append(int(h.response))
        else:
            reviewed_facts += 1
            review_known.append(int(h.response))
        total_seen += 1
        overall_known.append(int(h.response))

        if h.elapsed_milliseconds_text is None:
            if h.elapsed_seconds_text is not None:
                elapsed_milliseconds_text += 1000 * h.elapsed_seconds_text
        else:
            elapsed_milliseconds_text += h.elapsed_milliseconds_text
        if h.elapsed_milliseconds_answer is None:
            if h.elapsed_seconds_answer is not None:
                elapsed_milliseconds_answer += 1000 * h.elapsed_seconds_answer
        else:
            elapsed_milliseconds_answer += h.elapsed_milliseconds_answer

    new_known_rate = 0 if len(new_known) == 0 else np.mean(new_known)
    review_known_rate = 0 if len(review_known) == 0 else np.mean(review_known)
    known_rate = 0 if len(overall_known) == 0 else np.mean(overall_known)
    total_milliseconds = elapsed_milliseconds_text + elapsed_milliseconds_answer

    t2 = datetime.now()

    return {
        'new_facts': new_facts,
        'reviewed_facts': reviewed_facts,
        'total_seen': total_seen,
        'total_milliseconds': total_milliseconds,
        'total_seconds': total_milliseconds // 1000,
        'total_minutes': total_milliseconds // 60000,
        'elapsed_milliseconds_text': elapsed_milliseconds_text,
        'elapsed_milliseconds_answer': elapsed_milliseconds_answer,
        'elapsed_seconds_text': elapsed_milliseconds_text // 1000,
        'elapsed_seconds_answer': elapsed_milliseconds_answer // 1000,
        'elapsed_minutes_text': elapsed_milliseconds_text // 60000,
        'elapsed_minutes_answer': elapsed_milliseconds_answer // 60000,
        'known_rate': round(known_rate * 100, 2),
        'new_known_rate': round(new_known_rate * 100, 2),
        'review_known_rate': round(review_known_rate * 100, 2),
        'profile_get_history_records': (t1 - t0).total_seconds(),
        'profile_compute_stats': (t2 - t1).total_seconds(),
        'profile_n_history_records': len(history_records),
    }


class Ranking(BaseModel):
    user_id: int
    rank: int
    value: Union[int, float]


class Leaderboard(BaseModel):
    leaderboard: List[Ranking]
    total: int
    rank_type: str
    user_place: Optional[int] = None
    user_id: Optional[str] = None
    skip: Optional[int] = 0
    limit: Optional[int] = None


@app.get('/api/karl/leaderboard')
@cached(cache=TTLCache(maxsize=1024, ttl=1800))
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
    scheduler = schedulers[env]

    stats = {}
    for user in scheduler.db.query(User):
        if not user.user_id.isdigit():
            continue

        stats[user.user_id] = get_user_stats(
            user_id=user.user_id,
            env=env,
            deck_id=deck_id,
            date_start=date_start,
            date_end=date_end
        )

    profile_keys = [
        'profile_get_history_records',
        'profile_compute_stats',
        'profile_n_history_records',
    ]
    profile = {
        key: np.sum([v[key] for v in stats.values()])
        for key in profile_keys
    }

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

    pprint(profile)
    return leaderboard


@atexit.register
def finalize_db():
    for scheduler in schedulers.values():
        scheduler.db.commit()
        scheduler.db.close()
