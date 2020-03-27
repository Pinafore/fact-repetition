#!/usr/bin/env python
# coding: utf-8

import atexit
from fastapi import FastAPI
from typing import List
from datetime import datetime
from pydantic import BaseModel

from util import Flashcard, Params
from scheduler import MovingAvgScheduler


app = FastAPI()
scheduler = MovingAvgScheduler()


@app.post('/api/karl/predict')
def karl_predict(card: Flashcard):
    score = scheduler.predict_one(card.dict())
    return {
        'prob': score
    }

@app.post('/api/karl/schedule')
def karl_schedule(cards: List[Flashcard]):
    for i, _ in enumerate(cards):
        cards[i] = cards[i].dict()
        if cards[i]['date'] is None:
            cards[i]['date'] = str(datetime.now())
    return scheduler.schedule(cards)

@app.post('/api/karl/update')
def karl_update(cards: List[Flashcard]):
    # add date to card if missing
    for i, _ in enumerate(cards):
        cards[i] = cards[i].dict()
        if cards[i]['date'] is None:
            cards[i]['date'] = str(datetime.now())
    return scheduler.update(cards)


class UserID(BaseModel):
    user_id: str = None


@app.post('/api/karl/reset')
def karl_reset(user_id: UserID):
    scheduler.reset(user_id=user_id.dict().get('user_id', None))

@app.post('/api/karl/set_params')
def karl_set_params(params: Params):
    scheduler.set_params(params.dict())

@app.post('/api/karl/status')
def karl_status():
    return True

@atexit.register
def finalize_db():
    scheduler.db.finalize()
