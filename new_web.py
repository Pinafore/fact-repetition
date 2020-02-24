import os
import json
import pickle
import numpy as np
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI
from abc import ABC, abstractmethod

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from whoosh.fields import *
from whoosh.index import create_in
from whoosh.index import open_dir
from whoosh.qparser import QueryParser

from fact.util import Flashcard

diagnostic_file = 'data/diagnostic_flashcards.pkl'
records_file = 'data/jeopardy_310326_question_player_pairs_20190612.pkl'
questions_file = 'data/jeopardy_358974_questions_20190612.pkl'



class Scheduler(ABC):

    @abstractmethod
    def reset(self):
        # reset model and return init state
        pass

    @abstractmethod
    def predict(self, cards):
        # estimate difficulty of cards
        pass
    
    @abstractmethod
    def embed(self, cards):
        # embed card in topic space. return augmented cards
        pass

    @abstractmethod
    def set_params(self, params):
        pass


class MovingAvgScheduler(Scheduler):

    def __init__(self, n_components=20,
                 lambda_qrep=0.1, lambda_category=-0.3,
                 lambda_repetition=-1.0, lambda_leitner=1.0,
                 step_correct=0.5, step_wrong=0.05, step_qrep=0.3,
                 vectorizer_path='checkpoints/tf_vectorizer.pkl',
                 lda_path='checkpoints/lda.pkl'):
        self.n_components = n_components
        self.lambda_qrep = lambda_qrep
        self.lambda_category = lambda_category
        self.lambda_repetition = lambda_repetition
        self.lambda_leitner = lambda_leitner
        self.step_correct = step_correct
        self.step_wrong = step_wrong
        self.step_qrep = step_qrep
        self.vectorizer_path = vectorizer_path
        self.lda_path = lda_path

        print('loading question and records...')
        with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
            records_df = pickle.load(f)
        with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
            questions_df = pickle.load(f)
            # questions_df = questions_df.rename(columns={
            #     'gameid': 'game_id',
            #     'questionid': 'question_id',
            #     'airdate': 'air_date',
            #     'catnum': 'cat_num',
            #     'clue': 'text',
            # })
        karl_to_question_id = questions_df.to_dict()['question_id']
            
        # augment question_df with record stats
        print('merging dfs...')
        df = questions_df.set_index('question_id').join(records_df.set_index('question_id'))
        df = df[df['correct'].notna()]
        df['correct'] = df['correct'].apply(lambda x: 1 if x == 1 else 0)
        df_grouped = df.reset_index()[['question_id', 'correct']].groupby('question_id')
        dict_correct_mean = df_grouped.mean().to_dict()['correct']
        dict_records_cnt = df_grouped.count().to_dict()['correct']
        questions_df['accuracy'] = questions_df['question_id'].apply(lambda x: dict_correct_mean.get(x, None))
        questions_df['count'] = questions_df['question_id'].apply(lambda x: dict_records_cnt.get(x, None))
        self.questions_df = questions_df
        self.question_id_set = set(self.questions_df['question_id'])

        index_dir = 'whoosh_index'
        if not os.path.exists(index_dir):
            print('building whoosh...')
            self.build_whoosh()

        if not (os.path.exists(self.vectorizer_path) \
                and os.path.exists(self.lda_path)):
            with open(diagnostic_file, 'rb') as f:
                diagnostic_cards = pickle.load(f)
            print('building lda...')
            build_lda(cards)
        
        with open(self.vectorizer_path, 'rb') as f:
            self.tf_vectorizer = pickle.load(f)
        with open(self.lda_path, 'rb') as f:
            self.lda = pickle.load(f)

        self.reset()

    def reset(self):
        # round number since start
        self.round_num = 0
        # topical representation
        self.qrep = np.array([1 / n_components for _ in range(n_components)])
        # average difficulty (user accuracy) of questions seen so far
        # TODO replace 0.5 with average user accuracy
        self.prob = [0.5 for _ in range(n_components)],
        # TODO might need to remove for quizbowl?
        self.category = 'HISTORY'

        # cache of embeddings
        self.embed_cache = dict()
        # cache of card probability
        self.prob_cache = dict()
        # cache of previous study round number
        self.prev_cache = dict()

    def build_lda(self, cards):
        texts = [card['text'] for card in cards]
        tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                        max_features=50000,
                                        stop_words='english')
        tf = tf_vectorizer.fit_transform(texts)
        lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)
        # tf_feature_names = tf_vectorizer.get_feature_names()
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(tf_vectorizer, f)
        with open(self.lda_path, 'wb') as f:
            pickle.dump(lda, f)

    def build_whoosh(self):
        # TODO
        pass

    def embed(self, cards):
        return [self.embed_one(card) for card in cards]

    def embed_one(self, card):
        if 'embedding' in card:
            self.embed_cache[card['question_id']] = card['embedding']
            return card
        if card['question_id'] in self.embed_cache:
            card['embedding'] = self.embed_cache[card['question_id']]
            return card
        embedding = lda.transform(tf_vectorizer.transform([card['text']]))
        self.embed_cache[card['question_id']] = embedding
        card['embedding'] = embedding
        return card

    def predict(self, cards):
        return [self.predict_one(card) for card in cards]

    def retrieve(self, card):
        record_id = self.karl_to_question_id[int(card['question_id'])]

        # 1. try to find in records with gameid-catnum-level
        if record_id in self.question_id_set:
            hits = self.questions_df[self.questions_df.question_id == record_id]
            cards = [card.to_dict() for idx, card in hits.iterrows()]
            return cards, [1 / len(hits) for _ in range(len(hits))]

        # 2. do text search
        
    def predict_one(self, card):
        if 'prob' in card:
            self.prob_cache[card['question_id']] = card['prob']
            return card
        if card['question_id'] in self.prob_cache:
            return self.prob_cache[card['question_id']]
        prob = self._predict_one(card)
        self.prob_cache[card['question_id']] = prob
        card['prob'] = prob
        return card

    def _predict_one(self, card):
        # 1. find same or similar card in records
        cards, weights = self.retrieve(card)
        if len(cards) > 0:
            return np.dot([x['accuracy'] for x in cards], weights)
        # 2. use model to predict

    def set_params(self, params):
        pass
            
    
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


app = FastAPI()

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


if __name__ == '__main__':
    scheduler = MovingAvgScheduler()
