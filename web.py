from pydantic import BaseModel
from fastapi import FastAPI
from typing import List, Optional

from scheduler import MovingAvgScheduler


class Flashcard(BaseModel):
    text: str
    user_id: Optional[str]
    question_id: Optional[str]
    user_accuracy: Optional[float]
    user_buzzratio: Optional[float]
    user_count: Optional[float]
    question_accuracy: Optional[float]
    question_buzzratio: Optional[float]
    question_count: Optional[float]
    times_seen: Optional[float]
    times_correct: Optional[float]
    times_wrong: Optional[float]
    label: Optional[str]
    answer: Optional[str]
    category: Optional[str]


class Hyperparams(BaseModel):
    lambda_qrep: Optional[float]
    lambda_prob: Optional[float]
    lambda_category: Optional[float]
    lambda_repetition: Optional[float]
    lambda_leitner: Optional[float]
    step_correct: Optional[float]
    step_wrong: Optional[float]
    step_qrep: Optional[float]


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
    order, ranking = scheduler.schedule([x.dict() for x in cards])
    return {
        'order': order,
        'ranking': ranking,
        'user_qrep': scheduler.qrep,
        'user_prob': scheduler.prob,
    }

@app.post('/api/karl/update')
def karl_update(cards: List[Flashcard]):
    scheduler.update([x.dict() for x in cards])

@app.post('/api/karl/reset')
def karl_reset():
    scheduler.reset()

@app.post('/api/karl/set_hyperparameter')
def karl_set_params(params: Hyperparams):
    scheduler.set_params(params.dict())
