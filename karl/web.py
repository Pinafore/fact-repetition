#!/usr/bin/env python
# coding: utf-8

import json
import atexit
from fastapi import FastAPI
from typing import List
from datetime import datetime

from karl.util import ScheduleRequest, Params, parse_date
from karl.scheduler import MovingAvgScheduler

app = FastAPI()
scheduler = MovingAvgScheduler()

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
    date = datetime.now()
    if requests[0].date is not None:
        date = parse_date(requests[0].date)
    return scheduler.update(requests, date)

@app.post('/api/karl/set_params')
def set_params(params: Params):
    # TODO also pass a user_id
    scheduler.set_params(params)

@app.post('/api/karl/get_fact')
def get_fact(request: ScheduleRequest):
    return scheduler.get_fact(request).pack()

@app.get('/api/karl/reset_user/')
def reset_user(user_id: str = None):
    print('reset_user with user_id:', user_id)
    scheduler.reset_user(user_id=user_id)

@app.get('/api/karl/reset_fact/')
def reset_fact(fact_id: str = None):
    scheduler.reset_fact(fact_id=fact_id)

@app.get('/api/karl/status/')
def status():
    return True

@app.get('/api/karl/get_user/')
def get_user(user_id: str):
    return scheduler.get_user(user_id).pack()

@app.get('/api/karl/get_user_stats/')
def get_user_stats(user_id: str):
    user = scheduler.get_user(user_id).pack()
    return json.dumps(user.user_stats)

@atexit.register
def finalize_db():
    scheduler.db.finalize()
