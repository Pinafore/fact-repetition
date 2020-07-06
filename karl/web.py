#!/usr/bin/env python
# coding: utf-8

import atexit
import logging
import numpy as np
from fastapi import FastAPI
from typing import List
from datetime import datetime

from karl.util import ScheduleRequest, SetParams, Params, User, parse_date
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
    # NOTE assuming single user single date
    date = datetime.now()
    if len(requests) == 0:
        return {
            'order': [],
            'rationale': '<p>no fact received</p>',
            'facts_info': '',
        }

    logger.info(f'/karl/schedule with {len(requests)} facts and env={requests[0].env}')

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
    # NOTE assuming single user single date
    logger.info(f'/karl/update with {len(requests)} facts and env={requests[0].env}')

    date = datetime.now()
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
    users = schedulers[env].db.get_user()
    return [user.pack() for user in users]


@app.get('/api/karl/get_user_stats')
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
    scheduler = schedulers[env]

    history_records = scheduler.db.get_user_history(user_id, deck_id, date_start, date_end)

    new_facts = 0
    reviewed_facts = 0
    total_seen = 0
    total_seconds = 0
    new_known_rate = 0
    review_known_rate = 0
    elapsed_seconds_text = 0
    elapsed_seconds_answer = 0

    new_known, review_known, overall_known = [], [], []
    for h in history_records:
        user_snapshot = User.unpack(h.user_snapshot)
        if h.fact_id not in user_snapshot.previous_study:
            new_facts += 1
            new_known.append(int(h.response))
        else:
            reviewed_facts += 1
            review_known.append(int(h.response))
        total_seen += 1
        overall_known.append(int(h.response))

        total_seconds += h.elapsed_seconds_text
        elapsed_seconds_text += h.elapsed_seconds_text
        elapsed_seconds_answer += h.elapsed_seconds_answer

    new_known_rate = 0 if len(new_known) == 0 else np.mean(new_known)
    review_known_rate = 0 if len(review_known) == 0 else np.mean(review_known)
    known_rate = 0 if len(overall_known) == 0 else np.mean(overall_known)

    return {
        'new_facts': new_facts,
        'reviewed_facts': reviewed_facts,
        'total_seen': total_seen,
        'total_seconds': total_seconds,
        'total_minutes': total_seconds // 60,
        'elapsed_seconds_text': elapsed_seconds_text,
        'elapsed_seconds_answer': elapsed_seconds_answer,
        'elapsed_minutes_text': elapsed_seconds_text // 60,
        'elapsed_minutes_answer': elapsed_seconds_answer // 60,
        'known_rate': known_rate,
        'new_known_rate': new_known_rate,
        'review_known_rate': review_known_rate,
    }


@app.get('/api/karl/leaderboard')
def leaderboard(
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
    that ranks [skip + 1: skip + 1 + limit)
    '''
    env = 'dev' if env == 'dev' else 'prod'
    scheduler = schedulers[env]

    users = scheduler.db.get_user()
    stats = {}
    for user in users:
        if len(user.previous_study) < min_studied:
            continue

        stats[user.user_id] = get_user_stats(
            user_id=user.user_id,
            env=env,
            deck_id=deck_id,
            date_start=date_start,
            date_end=date_end
        )

    stats = sorted(stats.items(), key=lambda x: x[1][rank_type])[::-1]  # from high value to low
    return [
        {
            'user_id': user.user_id,
            'rank_type': rank_type,
            'value': v[rank_type],
        } for k, v in stats[skip: skip + limit + 1]
    ]


@atexit.register
def finalize_db():
    for scheduler in schedulers.values():
        scheduler.db.finalize()
