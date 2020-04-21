#!/usr/bin/env python
# coding: utf-8

import atexit
from fastapi import FastAPI
from typing import List
from datetime import datetime
from pydantic import BaseModel

from karl.util import ScheduleRequest, Params, parse_date
from karl.scheduler import MovingAvgScheduler

app = FastAPI()
scheduler = MovingAvgScheduler()

class UserID(BaseModel):
    user_id: str = None

@app.post('/api/karl/schedule')
def schedule(requests: List[ScheduleRequest]):
    # TODO assuming single user single date
    date = datetime.now()
    if requests[0].date is not None:
        date = parse_date(requests[0].date)
    return scheduler.schedule(requests, date)

@app.post('/api/karl/update')
def update(requests: List[ScheduleRequest]):

    # TODO assuming single user single date
    date = datetime.now()
    if requests[0].date is not None:
        date = parse_date(requests[0].date)
    return scheduler.update(requests, date)

@app.post('/api/karl/reset')
def reset(user_id: UserID):
    user_id = user_id.dict().get('user_id', None)
    scheduler.reset(user_id=user_id)

@app.post('/api/karl/set_params')
def set_params(params: Params):
    scheduler.set_params(params)

@app.post('/api/karl/status')
def status():
    return True

@app.post('/api/karl/get_user')
def get_user(user_id: UserID):
    user_id = user_id.dict()['user_id']
    return scheduler.get_user(user_id).pack()

@app.post('/api/karl/get_card')
def get_card(request: ScheduleRequest):
    return scheduler.get_card(request).pack()

@atexit.register
def finalize_db():
    scheduler.db.finalize()
