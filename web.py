#!/usr/bin/env python
# coding: utf-8

import atexit
from fastapi import FastAPI
from typing import List
from datetime import datetime
from pydantic import BaseModel

from util import ScheduleRequest, Params, parse_date
from scheduler import MovingAvgScheduler

app = FastAPI()
scheduler = MovingAvgScheduler()

@app.post('/api/karl/schedule')
def karl_schedule(requests: List[ScheduleRequest]):
    # TODO assuming single user single date
    date = datetime.now()
    if requests[0].date is not None:
        date = parse_date(requests[0].date)
    return scheduler.schedule(requests, date)

@app.post('/api/karl/update')
def karl_update(requests: List[ScheduleRequest]):

    # TODO assuming single user single date
    date = datetime.now()
    if requests[0].date is not None:
        date = parse_date(requests[0].date)
    return scheduler.update(requests, date)

class UserID(BaseModel):
    user_id: str = None

@app.post('/api/karl/reset')
def karl_reset(user_id: UserID):
    user_id = user_id.dict().get('user_id', None)
    scheduler.reset(user_id=user_id)

@app.post('/api/karl/set_params')
def karl_set_params(params: Params):
    scheduler.set_params(params)

@app.post('/api/karl/status')
def karl_status():
    return True

@atexit.register
def finalize_db():
    scheduler.db.finalize()
