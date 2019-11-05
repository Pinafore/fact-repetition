from typing import Optional
from starlette.responses import Response
from pydantic import BaseModel
from fastapi import FastAPI

from fact.karl_predictor import KarlPredictor, KARL_PREDICTOR

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


ARCHIVE_PATH = 'models/karl-rnn'

predictor  = KarlPredictor.from_path(
    ARCHIVE_PATH, predictor_name=KARL_PREDICTOR
)
app = FastAPI()

@app.post('/api/karl/predict')
def karl_predict(flashcard: Flashcard):
    pred = predictor.predict_json(flashcard.dict())
    return {'probs': pred['probs']}
