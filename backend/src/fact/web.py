import torch
import numpy as np

from allennlp.nn import util
from allennlp.data.dataset import Batch

from typing import List, Optional
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
    label: Optional[str]


ARCHIVE_PATH = 'models/karl-rnn'

predictor = KarlPredictor.from_path(
    ARCHIVE_PATH, predictor_name=KARL_PREDICTOR)
learning_rate = 1e-4
optimizer = torch.optim.Adam(predictor._model.parameters(), lr=learning_rate)

UPDATE_EPOCHS = 2

app = FastAPI()

@app.post('/api/karl/predict')
def karl_predict(flashcard: Flashcard):
    pred = predictor.predict_json(flashcard.dict())
    return {'probs': pred['probs'], 'all_labels': pred['all_labels']}


@app.post('/api/karl/schedule')
def karl_schedule(flashcards: List[Flashcard]):
    '''
    query the retention model on a set of flashcards, return the order of those
    cards to be studied. flashcard with closest to 0.5 retention probability is
    studied first.
    '''
    preds = [predictor.predict_json(card.dict()) for card in flashcards]
    probs = [x['probs'][0] for x in preds]  # labels = [correct, wrong]
    card_order = np.argsort(np.abs(0.5 - np.asarray(probs))).tolist()
    return {
        'probs': [x['probs'] for x in preds],
        'all_labels': [x['all_labels'] for x in preds],
        'card_order': card_order
    }


@app.post('/api/karl/update')
def karl_update(flashcards: List[Flashcard]):
    '''
    update the retention model using user study records
    each card should have a 'label' either 'correct' or 'wrong'
    '''
    flashcards = [card.dict() for card in flashcards]
    instances = predictor._batch_json_to_instances(flashcards)
    batch_size = len(instances)
    cuda_device = predictor._model._get_prediction_device()
    dataset = Batch(instances)
    dataset.index_instances(predictor._model.vocab)
    model_input = util.move_to_device(dataset.as_tensor_dict(), cuda_device)
    for i in range(UPDATE_EPOCHS):
        predictor._model.zero_grad()
        outputs = predictor._model(**model_input)
        outputs['loss'].backward()
        optimizer.step()
    # trainer._save_checkpoint("{0}.{1}".format(epoch, training_util.time_to_str(int(last_save_time))))
