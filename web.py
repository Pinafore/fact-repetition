from fastapi import FastAPI
from typing import List
from datetime import datetime

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
    order, ranking, rationale = scheduler.schedule(cards)
    return {
        'order': order,
        'ranking': ranking,
        'rationale': rationale
    }

@app.post('/api/karl/update')
def karl_update(cards: List[Flashcard]):
    # add date to card if missing
    for i, _ in enumerate(cards):
        cards[i] = cards[i].dict()
        if cards[i]['date'] is None:
            cards[i]['date'] = str(datetime.now())
    scheduler.update(cards)

@app.post('/api/karl/reset')
def karl_reset():
    scheduler.reset()

@app.post('/api/karl/set_hyperparameter')
def karl_set_params(params: Params):
    scheduler.set_params(params.dict())
