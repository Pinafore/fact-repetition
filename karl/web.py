#!/usr/bin/env python
# coding: utf-8

import atexit
import logging
import numpy as np
from fastapi import FastAPI
from typing import List
from datetime import datetime

from karl.util import ScheduleRequest, Params, User, parse_date
from karl.scheduler import MovingAvgScheduler


app = FastAPI()
scheduler = MovingAvgScheduler(db_filename='db.sqlite.dev')


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


def update_fact_id_with_env(requests: List[ScheduleRequest]):
    for request in requests:
        if request.env is not None:
            if request.env == 'dev':
                request.fact_id = f'dev_{request.fact_id}'
            else:
                request.fact_id = f'prod_{request.fact_id}'
    return requests


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

    logger.info('/karl/schedule with {} requests in {}'.format(len(requests), requests[0].env))
    if requests[0].date is not None:
        date = parse_date(requests[0].date)
    requests = update_fact_id_with_env(requests)
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
    logger.info('/karl/update with {} requests'.format(len(requests)))
    date = datetime.now()
    if requests[0].date is not None:
        date = parse_date(requests[0].date)
    requests = update_fact_id_with_env(requests)
    return scheduler.update(requests, date)


@app.post('/api/karl/set_params')
def set_params(params: Params):
    scheduler.set_user_params(params)


@app.post('/api/karl/get_fact')
def get_fact(request: ScheduleRequest):
    if request.env is not None:
        request.fact_id = '{}_{}'.format(request.env, request.fact_id)
    return scheduler.get_fact(request).pack()


@app.get('/api/karl/reset_user/{user_id}')
def reset_user(user_id: str = None):
    scheduler.reset_user(user_id=user_id)


@app.get('/api/karl/reset_fact/{fact_id}')
def reset_fact(fact_id: str = None):
    scheduler.reset_fact(fact_id=fact_id)


@app.get('/api/karl/status')
def status():
    return True


@app.get('/api/karl/get_user/{user_id}')
def get_user(user_id: str):
    return scheduler.get_user(user_id).pack()


@app.get('/api/karl/get_all_users')
def get_all_users():
    users = scheduler.db.get_user()
    return [user.pack() for user in users]


@app.get('/api/karl/get_user_stats/')
def get_user_stats(user_id: str, date_start: str = None, date_end: str = None):
    '''
    Retrieve user stats within given date span.
    new_facts: int
    reviewed_facts: int
    total_seen: int
    total_seconds: int
    new_known_rate: float
    review_known_rate: float
    '''
    history_records = scheduler.db.get_user_history(user_id, date_start, date_end)

    new_facts = 0
    reviewed_facts = 0
    total_seen = 0
    total_seconds = 0
    new_known_rate = 0
    review_known_rate = 0
    new_known, review_known = [], []
    for h in history_records:
        user_snapshot = User.unpack(h.user_snapshot)
        if h.fact_id not in user_snapshot.previous_study:
            new_facts += 1
            new_known.append(int(h.response))
        else:
            reviewed_facts += 1
            review_known.append(int(h.response))
        total_seen = len(user_snapshot.previous_study)
        total_seconds += h.elapsed_seconds_text

    new_known_rate = 0 if len(new_known) == 0 else np.mean(new_known)
    review_known_rate = 0 if len(review_known) == 0 else np.mean(review_known)

    return {
        'new_facts': new_facts,
        'reviewed_facts': reviewed_facts,
        'total_seen': total_seen,
        'total_seconds': total_seconds,
        'new_known_rate': new_known_rate,
        'review_known_rate': review_known_rate,
    }


@atexit.register
def finalize_db():
    scheduler.db.finalize()
