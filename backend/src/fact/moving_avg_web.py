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
    alpha: Optional[float]
    beta: Optional[int]
    lambd: Optional[float]
    category_hyper: Optional[float]


n_features = 50000
n_components = 20
with open('tf_vectorizer.pkl', 'rb') as f:
    tf_vectorizer = pickle.load(f)
with open('lda.pkl', 'rb') as f:
    lda = pickle.load(f)

curr_qrep = np.array([1 / n_components for _ in range(n_components)])
curr_prob = 0.5
curr_cate = 'HISTORY'
alpha = 0.1 # prob
beta = 0.1 # qrep
lambd = 0.7 # dist = lambda * dist_qrep + (1 - lambda) * dist_prob
category_hyper = 0.3
cached_qrep = dict()
cached_prob = dict()

app = FastAPI()

df = pickle.load(open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb'))
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
    if card['question_id'] in cached_qrep:
        return cached_qrep[card['question_id']]
    qrep = lda.transform(tf_vectorizer.transform([card['text']]))
    cached_qrep[card['question_id']] = qrep
    return qrep


def get_prob(card):
    if 'prob' in card:
        return card['prob']
    if card['question_id'] in cached_prob:
        return cached_prob[card['question_id']]
    qid = karl_to_question_id[int(card['question_id'])]
    prob = (df[df.question_id == qid]['correct'] / 2 + 0.5).mean()
    cached_prob[card['question_id']] = prob
    return prob


@app.post('/api/karl/predict')
def karl_predict(flashcard: Flashcard):
    return curr_prob


@app.post('/api/karl/schedule')
def karl_schedule(flashcards: List[Flashcard]):
    flashcards = [x.dict() for x in flashcards]
    for i, card in enumerate(flashcards):
        flashcards[i]['qrep'] = get_qrep(card)
        flashcards[i]['prob'] = get_prob(card)
        a, b = curr_qrep, card['qrep']
        dist_x = 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        dist_y = card['prob'] - curr_prob
        flashcards[i]['dist_qrep'], flashcards[i]['dist_prob'] = dist_x, dist_y
        flashcards[i]['dist'] = lambd * dist_x + (1 - lambd) * np.abs(dist_y)
        flashcards[i]['dist'] -= category_hyper if flashcards[i]['category'] == curr_cate else 0

    reverse_index = {x['question_id']: i for i, x in enumerate(flashcards)}

    cards_more_difficult, cards_less_difficult = [], []
    for card in flashcards:
        if card['dist_prob'] < 0:
            cards_more_difficult.append(card)
        else:
            cards_less_difficult.append(card)
    cards_more_difficult = sorted(cards_more_difficult, key=lambda x: x['dist'])
    cards_less_difficult = sorted(cards_less_difficult, key=lambda x: x['dist'])
    cards_sorted = cards_more_difficult + cards_less_difficult
    card_order = [reverse_index[x['question_id']] for x in cards_sorted]
    return {
        'all_labels': [['correct', 'wrong'] for x in flashcards],
        'card_order': card_order,
        'curr_prob': curr_prob,
        'probs': [x['prob'] for x in flashcards],
        # curr_qrep': curr_qrep,
        # 'dist_prob': cards_sorted[0]['dist_prob'],
        # 'dist_qrep': cards_sorted[0]['dist_qrep'],
    }


@app.post('/api/karl/update')
def karl_update(flashcards: List[Flashcard]):
    global curr_prob, curr_qrep, curr_cate 
    flashcards = [x.dict() for x in flashcards]
    for card in flashcards:
        card['prob'], card['qrep'] = get_prob(card), get_qrep(card)
        if card['label'] == 'correct':
            curr_prob = alpha * card['prob'] + (1 - alpha) * curr_prob
        else:
            curr_prob += 0.05
        curr_prob = min(1.0, curr_prob)
        curr_qrep = beta * card['qrep'] + (1 - beta) * card['qrep']
        curr_cate = card['category']


@app.post('/api/karl/reset')
def karl_reset():
    global curr_qrep, curr_prob, curr_cate
    curr_qrep = np.array([1 / n_components for _ in range(n_components)])
    curr_prob = 0.5
    curr_cate = 'HISTORY'


@app.post('/api/karl/set_hyperparameter')
def karl_set_hyperparameter(params: Hyperparams):
    '''
    update the retention model using user study records
    each card should have a 'label' either 'correct' or 'wrong'
    '''
    global alpha, beta, lambd, category_hyper
    params = params.dict()
    alpha = params.get('alpha', alpha)
    beta = params.get('beta', beta)
    lambd = params.get('lambda', lambd)
    category_hyper = params.get('category_hyper', category_hyper)
    return {
        'alpha': alpha,
        'beta': beta,
        'lambda': lambd,
        'category_hyper': category_hyper
    }
