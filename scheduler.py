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
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser


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

    @abstractmethod
    def schedule(self, cards):
        pass

    @abstractmethod
    def update(self, cards):
        pass


class MovingAvgScheduler(Scheduler):

    def __init__(self, n_components=20,
                 lambda_qrep=0.1, lambda_prob=0.7, lambda_category=0.3,
                 lambda_repetition=-1.0, lambda_leitner=1.0,
                 step_correct=0.5, step_wrong=0.05, step_qrep=0.3,
                 vectorizer_path='checkpoints/tf_vectorizer.pkl',
                 lda_path='checkpoints/lda.pkl'):
        self.n_components = n_components
        self.lambda_qrep = lambda_qrep
        self.lambda_prob = lambda_prob
        self.lambda_category = lambda_category
        self.lambda_repetition = lambda_repetition
        self.lambda_leitner = lambda_leitner
        self.step_correct = step_correct
        self.step_wrong = step_wrong
        self.step_qrep = step_qrep
        self.vectorizer_path = vectorizer_path
        self.lda_path = lda_path

        self.diagnostic_file = 'data/diagnostic_questions.pkl'
        self.index_dir = 'whoosh_index'

        print('loading question and records...')
        with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
            records_df = pickle.load(f)
        with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
            questions_df = pickle.load(f)
        self.karl_to_question_id = questions_df.to_dict()['question_id']
            
        # augment question_df with record stats
        print('merging dfs...')
        df = questions_df.set_index('question_id').join(records_df.set_index('question_id'))
        df = df[df['correct'].notna()]
        df['correct'] = df['correct'].apply(lambda x: 1 if x == 1 else 0)
        df_grouped = df.reset_index()[['question_id', 'correct']].groupby('question_id')
        dict_correct_mean = df_grouped.mean().to_dict()['correct']
        dict_records_cnt = df_grouped.count().to_dict()['correct']
        questions_df['prob'] = questions_df['question_id'].apply(lambda x: dict_correct_mean.get(x, None))
        questions_df['count'] = questions_df['question_id'].apply(lambda x: dict_records_cnt.get(x, None))
        self.questions_df = questions_df
        self.question_id_set = set(self.questions_df['question_id'])

        if not os.path.exists(self.index_dir):
            print('building whoosh...')
            self.build_whoosh()
        self.ix = open_dir(self.index_dir)

        if not (os.path.exists(self.vectorizer_path) \
                and os.path.exists(self.lda_path)):
            with open(self.diagnostic_file, 'rb') as f:
                diagnostic_cards = pickle.load(f)
            print('building lda...')
            self.build_lda(diagnostic_cards)
        
        with open(self.vectorizer_path, 'rb') as f:
            self.tf_vectorizer = pickle.load(f)
        with open(self.lda_path, 'rb') as f:
            self.lda = pickle.load(f)

        with open(self.diagnostic_file, 'rb') as f:
            diagnostic_cards = pickle.load(f)
        self.reset(self.estimate_avg(diagnostic_cards))
        print('scheduler ready')

    def estimate_avg(self, cards):
        # estimate the average acccuracy for each component
        # use for initializing user estimate
        estimate_file = 'data/diagnostic_avg_estimate.txt'
        if os.path.exists(estimate_file):
            with open(estimate_file) as f:
                return [float(x) for x in f.readlines()]

        texts = [x['text'] for x in cards]
        qreps = self.lda.transform(self.tf_vectorizer.transform(texts))
        estimates = [[] for _ in range(self.n_components)]
        for card, qrep in zip(cards, qreps):
            topic_idx = np.argmax(qrep)
            prob = self.predict_one(card)
            estimates[topic_idx].append(prob)
        estimates = [np.mean(x) for x in estimates]
        with open('data/diagnostic_avg_estimate.txt', 'w') as f:
            for e in estimates:
                f.write(str(e) + '\n')
        return estimates

    def reset(self, prob=None):
        # round number since start
        self.round_num = 0
        # topical representation
        self.qrep = np.array([1 / self.n_components for _ in range(self.n_components)])
        # average difficulty (user accuracy) of questions seen so far
        # TODO replace 0.5 with average user accuracy
        if prob is None:
            prob = [0.5 for _ in range(self.n_components)]
        self.prob = prob
        # TODO might need to remove for quizbowl?
        self.category = 'HISTORY'

        # cache of embeddings
        self.qrep_cache = dict()
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
        lda = LatentDirichletAllocation(n_components=self.n_components,
                                        max_iter=5, learning_method='online',
                                        learning_offset=50.,
                                        random_state=0)
        lda.fit(tf)
        # tf_feature_names = tf_vectorizer.get_feature_names()
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(tf_vectorizer, f)
        with open(self.lda_path, 'wb') as f:
            pickle.dump(lda, f)

    def build_whoosh(self):
        if not os.path.exists(self.index_dir):
            os.mkdir(self.index_dir)
        schema = Schema(
            question_id=ID(stored=True),
            text=TEXT(stored=True),
            answer=TEXT(stored=True)
        )
        ix = create_in(self.index_dir, schema)
        writer = ix.writer()
        
        for idx, q in tqdm(self.questions_df.iterrows()):
            writer.add_document(
                question_id=q['question_id'],
                text=q['text'],
                answer=q['answer']
            )
        writer.commit()

    def embed(self, cards):
        for i, card in enumerate(cards):
            cards[i]['qrep'] = self.embed_one(card)
        # TODO potential speed up here: batch cards without qrep
        return cards

    def embed_one(self, card):
        if 'qrep' in card:
            self.qrep_cache[card['question_id']] = card['qrep']
            return card['qrep']
        if card['question_id'] in self.qrep_cache:
            return self.qrep_cache[card['question_id']]
        qrep = self.lda.transform(self.tf_vectorizer.transform([card['text']]))
        self.qrep_cache[card['question_id']] = qrep
        return qrep

    def predict(self, cards):
        for i, card in enumerate(cards):
            cards[i]['prob'] = self.predict_one(card)
        return cards

    def retrieve(self, card):
        record_id = self.karl_to_question_id[int(card['question_id'])]

        # 1. try to find in records with gameid-catnum-level
        if record_id in self.question_id_set:
            hits = self.questions_df[self.questions_df.question_id == record_id]
            if len(hits) > 0:
                cards = [card.to_dict() for idx, card in hits.iterrows()]
                return cards, [1 / len(hits) for _ in range(len(hits))]

        # 2. do text search
        with self.ix.searcher() as searcher:
            query = QueryParser("text", ix.schema).parse(card['text'])
            hits = searcher.search(query)
            hits = [x for x in hits if x['answer'] == card['answer']]
            if len(hits) > 0:
                scores = [x.score for x in hits]
                ssum = np.sum(scores)
                scores = [x / ssum for x in scores]
                cards = [self.questions_df[self.questions_df['question_id'] == x['question_id']].iloc[0] for x in hits]
                return cards, scores

        # 3. not found
        return [], []
        
    def predict_one(self, card):
        if 'prob' in card:
            self.prob_cache[card['question_id']] = card['prob']
            return card['prob']
        if card['question_id'] in self.prob_cache:
            return self.prob_cache[card['question_id']]
        prob = self._predict_one(card)
        self.prob_cache[card['question_id']] = prob
        return prob

    def _predict_one(self, card):
        # 1. find same or similar card in records
        cards, scores = self.retrieve(card)
        if len(cards) > 0:
            return np.dot([x['prob'] for x in cards], scores)
        # 2. use model to predict
        # TODO
        return 0

    def set_params(self, params):
        self.lambda_qrep = params.get('lambda_qrep', lambda_qrep)
        self.lambda_prob = params.get('lambda_prob', lambda_prob)
        self.lambda_category = params.get('lambda_category', lambda_category)
        self.lambda_repetition = params.get('lambda_repetition', lambda_repetition)
        self.lambda_leitner = params.get('lambda_leitner', lambda_leitner)
        self.step_correct = params.get('step_correct', step_correct)
        self.step_wrong = params.get('step_wrong', step_wrong)
        self.step_qrep = params.get('step_qrep', step_qrep)

    def dist_rep(self, card):
        # distance penalty due to repetition
        prev = self.prev_cache.get(card['question_id'], self.round_num)
        return self.round_num - prev
    
    def dist_category(self, card):
        return int(card['category'] != self.category)
    
    def dist_prob(self, card):
        if 'qrep' not in card:
            card['qrep'] = self.embed_one(card)
        topic_idx = np.argmax(card['qrep'])
        d = card['prob'] - self.prob[topic_idx]
        # penalize easier questions
        d *= 1 if d < 0 else 10
        return abs(d)
    
    def dist_qrep(self, card):
        def cosine_distance(a, b):
            return 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cosine_distance(self.qrep, card['qrep'])

    def dist_leitner(self, card):
        # TODO
        pass
    
    def score_one(self, card):
        # compute distance metric of given card to current state
        # dist = topic repr dist + topical difficulty dist + recency + leitner score
        # NOTE lower is better
        card['qrep'] = self.embed_one(card)
        card['prob'] = self.predict_one(card)
        return self.lambda_qrep * self.dist_qrep(card) \
            + self.lambda_prob * self.dist_prob(card) \
            + self.lambda_category * self.dist_category(card) \
            # + self.lambda_repetition * self.dist_rep(card)
            # + self.lambda_leitner * self.dist_leitner(card)

    def score(self, cards):
        cards = self.embed(self.predict(cards))
        return [self.score_one(card) for card in cards]
    
    def schedule(self, cards):
        scores = self.score(cards)
        for i, card in enumerate(cards):
            # NOTE lower is better
            cards[i]['score'] = scores[i]
        reverse_index = {x['question_id']: i for i, x in enumerate(cards)}
        cards_sorted = sorted(cards, key=lambda x: x['score'])
        # sorted indices 
        order = [reverse_index[x['question_id']] for x in cards_sorted]
        ranking = [order.index(i) for i, _ in enumerate(cards)]
        return order, ranking

    def update(self, cards):
        cards = self.embed(self.predict(cards))
        for card in cards:
            topic_index = np.argmax(card['qrep'])
            if card['label'] == 'correct':
                a = self.step_correct
                self.prob[topic_index] = a * card['prob'] + (1 - a) * self.prob[topic_index]
            else:
                self.prob[topic_index] += self.step_wrong
            self.prob[topic_index] = min(1.0, self.prob[topic_index])
            b = self.step_qrep
            self.qrep = b * card['qrep'] + (1 - b) * self.qrep
            self.category = card['category']
