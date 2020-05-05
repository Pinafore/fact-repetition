#!/usr/bin/env python
# coding: utf-8

import json
import atexit
import logging
from fastapi import FastAPI
from typing import List
from datetime import datetime

from karl.util import ScheduleRequest, Params, parse_date
from karl.scheduler import MovingAvgScheduler

app = FastAPI()
scheduler = MovingAvgScheduler()

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
    logger.info('/karl/schedule with {} requests'.format(len(requests)))
    date = datetime.now()
    if len(requests) == 0:
        return {
            'order': [],
            'rationale': '<p>no fact received</p>',
            'facts_info': '',
        }

    if requests[0].date is not None:
        date = parse_date(requests[0].date)

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
    return scheduler.update(requests, date)

@app.post('/api/karl/set_params')
def set_params(params: Params):
    scheduler.set_user_params(params)

@app.post('/api/karl/get_fact')
def get_fact(request: ScheduleRequest):
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

@app.get('/api/karl/get_user_stats/{user_id}')
def get_user_stats(user_id: str):
    user = scheduler.get_user(user_id)
    stats = user.user_stats.__dict__
    return json.dumps({
        'new_facts': stats['new_facts'],
        'reviewed_facts': stats['reviewed_facts'],
        'total_seen': stats['total_seen'],
        'total_seconds': stats['total_seconds'],
        'new_known_rate': stats['new_known_rate'],
        'review_known_rate': stats['review_known_rate'],
    })

@atexit.register
def finalize_db():
    scheduler.db.finalize()
