import pickle
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


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
    qrep: Optional[float]
    prob_difficult: Optional[float]
    prob_easy: Optional[float]
    category: Optional[float]
    repetition: Optional[float]
    leitner: Optional[float]
    lr_prob_correct: Optional[float]
    lr_prob_wrong: Optional[float]
    lr_qrep: Optional[float]


n_features = 50000
n_components = 20
with open('tf_vectorizer.pkl', 'rb') as f:
    tf_vectorizer = pickle.load(f)
with open('lda.pkl', 'rb') as f:
    lda = pickle.load(f)


def init_state(n_components):
    return {
        # round number since start
        'round': 0,
        # topical representation
        'qrep': np.array([1 / n_components for _ in range(n_components)]),
        # average difficulty (user accuracy) of questions seen so far
        # TODO: replace 0.5 with average user accuracy
        'prob': [0.5 for _ in range(n_components)],
        # TODO: might need to remove for quizbowl?
        'category': 'HISTORY'
    }

curr = init_state(n_components)
cache = {
    'qrep': dict(),
    'prob': dict(),
    'prev': dict(),
}

param = {
    'qrep': 0.1,
    'prob_difficult': 0.9,
    'prob_easy': 10,
    'category': -0.3,
    'repetition': -1.0,
    'leitner': 1.0,
    'lr_prob_correct': 0.5,
    'lr_prob_wrong': 0.05,
    'lr_qrep': 0.3,
}


app = FastAPI()

with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
    records_df = pickle.load(f)
with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
    karl_to_question_id = pickle.load(f).to_dict()['questionid']


def train_save_lda():
    with open('diagnostic_card_set.pkl', 'rb') as f:
        flashcards = pickle.load(f)
    all_text = [x['text'] for x in flashcards]
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                    max_features=n_features,
                                    stop_words='english')
    tf = tf_vectorizer.fit_transform(all_text)
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=0)
    lda.fit(tf)
    tf_feature_names = tf_vectorizer.get_feature_names()
    with open('tf_vectorizer.pkl', 'wb') as f:
        pickle.dump(tf_vectorizer, f)

    with open('lda.pkl', 'wb') as f:
        pickle.dump(lda, f)


def get_qrep(card):
    if 'qrep' in card:
        return card['qrep']
    if card['question_id'] in cache['qrep']:
        return cache['qrep'][card['question_id']]
    qrep = lda.transform(tf_vectorizer.transform([card['text']]))
    cache['qrep'][card['question_id']] = qrep
    return qrep


def get_prob(card):
    if 'prob' in card:
        return card['prob']
    if card['question_id'] in cache['prob']:
        return cache['prob'][card['question_id']]
    # use gameid-catnum-level to identify records
    record_id = karl_to_question_id[int(card['question_id'])]
    prob = (records_df[records_df.question_id == record_id]['correct'] / 2 + 0.5).mean()
    cache['prob'][card['question_id']] = prob
    # TODO: fall back to retention model if user record not available
    # TODO: handle quizbowl
    return prob


def get_dist_rep(card, param):
    # distance penalty due to repetition
    prev = cache['prev'].get(card['question_id'], 0)
    dist_rep = (curr['round'] - prev) * param['repetition']
    return dist_rep

def get_dist_category(card, param):
    return param['category'] if card['category'] == curr['category'] else 0

def cosine_distance(a, b):
    return 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

def get_dist_prob(card, param):
    topic_index = np.argmax(card['qrep'])
    dist_prob = card['prob'] - curr['prob'][topic_index]
    dist_prob *= param['prob_difficult'] if dist_prob < 0 else param['prob_easy']
    dist_prob = abs(dist_prob)
    return dist_prob

def get_dist_qrep(card, param):
    return param['qrep'] * cosine_distance(curr['qrep'], card['qrep'])

def get_dist(card):
    # compute distance metric of given card to current state
    # dist = topic repr dist + topical difficulty dist + recency + leitner score
    card['qrep'] = get_qrep(card)
    card['prob'] = get_prob(card)
    dist = get_dist_qrep(card, param) \
        + get_dist_prob(card, param) \
        + get_dist_category(card, param) \
        # + get_dist_rep(card, param)
    return dist
    

@app.post('/api/karl/predict')
def karl_predict(flashcard: Flashcard):
    # TODO: remove?
    return curr


@app.post('/api/karl/schedule')
def karl_schedule(flashcards: List[Flashcard]):
    flashcards = [x.dict() for x in flashcards]
    for i, card in enumerate(flashcards):
        flashcards[i]['dist'] = get_dist(card)
    reverse_index = {x['question_id']: i for i, x in enumerate(flashcards)}
    cards_sorted = sorted(flashcards, key=lambda x: x['dist'])
    card_order = [reverse_index[x['question_id']] for x in cards_sorted]
    return {
        'all_labels': [['correct', 'wrong'] for x in flashcards],
        'card_order': card_order,
        'curr_prob': curr['prob'],
    }


@app.post('/api/karl/update')
def karl_update(flashcards: List[Flashcard]):
    global curr
    flashcards = [x.dict() for x in flashcards]
    for card in flashcards:
        card['prob'] = get_prob(card)
        card['qrep'] = get_qrep(card)
        topic_index = np.argmax(card['qrep'])
        if card['label'] == 'correct':
            alpha = param['lr_prob_correct']
            curr['prob'][topic_index] = alpha * card['prob'] + (1 - alpha) * curr['prob'][topic_index]
        else:
            curr['prob'][topic_index] += param['lr_prob_wrong']
        curr['prob'][topic_index] = min(1.0, curr['prob'][topic_index])
        beta = param['lr_qrep']
        curr['qrep'] = beta * card['qrep'] + (1 - beta) * card['qrep']
        curr['category'] = card['category']


@app.post('/api/karl/reset')
def karl_reset():
    global curr
    curr = init_state(n_components)


@app.post('/api/karl/set_hyperparameter')
def karl_set_hyperparameter(params: Hyperparams):
    '''
    update the retention model using user study records
    each card should have a 'label' either 'correct' or 'wrong'
    '''
    global param
    new_param = params.dict()
    for name, value in param.items():
        param[name] = new_param.get(name, value)
    return param
