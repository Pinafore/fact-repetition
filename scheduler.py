#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import pickle
import numpy as np
import logging
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Tuple, Dict

import gensim
import en_core_web_lg

from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh.fields import Schema, ID, TEXT

from util import Params, Card, User, History, parse_date
from db import SchedulerDB

from build_lda import process
nlp = en_core_web_lg.load()
nlp.add_pipe(process, name='process', last=True)

logger = logging.getLogger('scheduler')

# current qrep is the discounted average of qreps of the past MAX_QREPS cards
MAX_QREPS = 10


class MovingAvgScheduler:

    def __init__(self, params: Params = None, db_filename='db.sqlite'):
        if params is None:
            params = Params()
        self.params = params
        self.db_filename = db_filename
        self.db = SchedulerDB(db_filename)

        logger.info('loading question and records...')
        with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
            records_df = pickle.load(f)
        with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
            questions_df = pickle.load(f)
        self.karl_to_question_id = questions_df.to_dict()['question_id']

        # augment question_df with record stats
        logger.info('merging dfs...')
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

        # setup whoosh for text search
        if not os.path.exists(self.params.whoosh_index):
            logger.info('building whoosh...')
            self.build_whoosh()
        self.ix = open_dir(self.params.whoosh_index)

        # LDA gensim
        self.lda = gensim.models.LdaModel.load(os.path.join(params.lda_dir, 'lda'))
        self.vocab = gensim.corpora.Dictionary.load_from_text(os.path.join(params.lda_dir, 'vocab.txt'))
        with open(os.path.join(params.lda_dir, 'topic_words.txt'), 'r') as f:
            self.topic_words = [l.strip() for l in f.readlines()]
        logger.info(self.topic_words)

        self.estimate_avg()

        logger.info('scheduler ready')

    def estimate_avg(self) -> np.ndarray:
        # estimate the average acccuracy for each component
        # use for initializing user estimate
        with open('data/diagnostic_questions.pkl', 'rb') as f:
            cards = pickle.load(f)
        estimate_file_dir = os.path.join(
            self.params.lda_dir, 'diagnostic_avg_estimate.txt')
        if os.path.exists(estimate_file_dir):
            logger.info('load user skill estimate')
            with open(estimate_file_dir) as f:
                return np.array([float(x) for x in f.readlines()])

        logger.info('estimate average user skill')
        qreps = self.embed([x['text'] for x in cards])
        estimates = [[] for _ in range(self.params.n_topics)]
        for card, qrep in zip(cards, qreps):
            topic_idx = np.argmax(qrep)
            prob = self.predict_one(card)
            estimates[topic_idx].append(prob)
        estimates = [np.mean(x) for x in estimates]
        with open(estimate_file_dir, 'w') as f:
            for e in estimates:
                f.write(str(e) + '\n')
        return np.array(estimates)

    def reset(self, user_id: str = None):
        # delete users and history from db
        self.db.delete_user(user_id=user_id)
        self.db.delete_history(user_id=user_id)

    def build_whoosh(self):
        if not os.path.exists(self.params.whoosh_index):
            os.mkdir(self.params.whoosh_index)
        schema = Schema(
            question_id=ID(stored=True),
            text=TEXT(stored=True),
            answer=TEXT(stored=True)
        )
        ix = create_in(self.params.whoosh_index, schema)
        writer = ix.writer()

        for idx, q in tqdm(self.questions_df.iterrows()):
            writer.add_document(
                question_id=q['question_id'],
                text=q['text'],
                answer=q['answer']
            )
        writer.commit()

    def get_card(self, card: dict) -> Card:
        '''get card from db, insert if new'''
        # retrieve from db if exists
        c = self.db.get_card(card['question_id'])
        if c is not None:
            return c
        # create new card and insert to db
        qrep = self.embed([card['text']])
        skill = np.zeros_like(qrep)
        skill[np.argmax(qrep)] = 1
        skill *= self.predict_one(card)
        date = card['date']
        new_card = Card(
            card_id=card['question_id'],
            text=card['text'],
            answer=card['answer'],
            qrep=qrep,
            skill=skill,
            category=card['category'],
            date=parse_date(date)
        )
        self.db.add_card(new_card)
        return new_card

    def embed(self, texts: List[str]) -> List[np.ndarray]:
        texts = (self.vocab.doc2bow(x) for x in nlp.pipe(texts))
        # need to set minimum_probability to a negative value to prevent output
        # skilping topics
        topics = self.lda.get_document_topics(texts, minimum_probability=-1)
        # TODO speed up?
        return [np.asarray([value for i, value in ts]) for ts in topics]

    def get_cards(self, cards: List[dict]) -> List[Card]:
        '''get cards from db, insert if new
        optimize speed by batch embedding
        '''
        to_be_embedded = []
        for i, card in enumerate(cards):
            card_id = card['question_id']
            cc = self.db.get_card(card_id)
            if cc is None:
                # where in the list to put this card back
                cards[i]['index'] = i
                to_be_embedded.append(card)
            else:
                cards[i] = cc

        if len(to_be_embedded) == 0:
            # all cards embedded & predicted
            return cards

        # batch embed and predict for speed up
        t0 = datetime.now()
        qreps = self.embed([x['text'] for x in to_be_embedded])
        t1 = datetime.now()
        probs = self.predict(to_be_embedded)
        t2 = datetime.now()
        for i, c in enumerate(to_be_embedded):
            # card skill is an one-hot vector
            # can be a distribution
            skill = np.zeros_like(qreps[i])
            skill[np.argmax(qreps[i])] = 1
            skill *= probs[i]

            new_card = Card(
                card_id=c['question_id'],
                text=c['text'],
                answer=c['answer'],
                qrep=qreps[i],
                skill=skill,
                category=c['category'],
                date=parse_date(c['date'])
            )

            # TODO batch add to db for speed up?
            self.db.add_card(new_card)
            # put it back
            assert cards[c['index']]['question_id'] == new_card.card_id
            cards[c['index']] = new_card
        t3 = datetime.now()
        logger.debug('************************')
        logger.debug('embed ' + str(t1 - t0))
        logger.debug('predict ' + str(t2 - t1))
        logger.debug('add to db ' + str(t3 - t2))
        return cards

    def get_user(self, user_id: str) -> User:
        '''get user from DB, insert if new'''
        # retrieve from db if exists
        u = self.db.get_user(user_id)
        if u is not None:
            return u

        # create new user and insert to db
        k = self.params.n_topics
        new_user = User(
            user_id=user_id,
            qrep=[np.array([1 / k for _ in range(k)])],
            skill=[self.estimate_avg()],
            category='HISTORY',
            date=datetime.now()
        )
        self.db.add_user(new_user)
        return new_user

    def retrieve(self, card: dict) -> Tuple[List[dict], List[float]]:
        record_id = self.karl_to_question_id[int(card['question_id'])]

        # 1. try to find in records with gameid-catnum-level
        if record_id in self.question_id_set:
            hits = self.questions_df[self.questions_df.question_id == record_id]
            if len(hits) > 0:
                cards = [card.to_dict() for idx, card in hits.iterrows()]
                return cards, [1 / len(hits) for _ in range(len(hits))]
        else:
            # TODO return better default value without text search
            return 0.5

        # 2. do text search
        with self.ix.searcher() as searcher:
            query = QueryParser("text", self.ix.schema).parse(card['text'])
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

    def predict_one(self, card: dict) -> float:
        return 0.5
        # # 1. find same or similar card in records
        # cards, scores = self.retrieve(card)
        # if len(cards) > 0:
        #     return np.dot([x['prob'] for x in cards], scores)
        # # TODO 2. use model to predict

    def predict(self, cards: List[dict]) -> List[float]:
        # TODO batch predict here?
        return [self.predict_one(card) for card in cards]

    def set_params(self, params: dict):
        self.params.__dict__.update(params)

    def dist_category(self, user: User, card: Card) -> float:
        if user.category is None or card.category is None:
            return 0
        return float(card.category.lower() != user.category.lower())

    def dist_skill(self, user: User, card: Card) -> float:
        # measures difficulty difference
        weight = 1
        skills, weights = [], []
        for q in user.skill:
            skills.append(weight * q)
            weights.append(weight)
            weight *= 0.9  # TODO use params instead of hard-coded
        skill = np.sum(skills, axis=0) / np.sum(weights)
        # skill is a (vector of) probability
        skill = np.clip(skill, a_min=0.0, a_max=1.0)
        topic_idx = np.argmax(card.qrep)
        d = card.skill[topic_idx] - skill[topic_idx]
        # penalize easier questions by ten
        d *= 1 if d <= 0 else 10
        return abs(d)

    def dist_time(self, user: User, card: Card) -> float:
        # cool down question for 10 minutes
        t = user.last_study_time.get(card.card_id, None)
        if t is None:
            return 0
        else:
            t1 = time.mktime(datetime.now().timetuple())
            t0 = time.mktime(t.timetuple())
            return max(10 - float(t1 - t0) / 60, 0)  # TODO use params instead of hard code

    def dist_qrep(self, user: User, card: Card) -> float:
        # measures topical similarity
        def cosine_distance(a, b):
            return 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        weight = 1
        qreps, weights = [], []
        for q in user.qrep:
            qreps.append(weight * q)
            weights.append(weight)
            weight *= self.params.decay_qrep
        qrep = np.sum(qreps, axis=0) / np.sum(weights)
        d = cosine_distance(qrep, card.qrep)
        # print('user qrep norm', np.linalg.norm(qrep))
        # print('card qrep norm', np.linalg.norm(card.qrep))
        # print('dot product', qrep @ card.qrep)
        # print('normalized', 1 - d)
        # print('cosine distance', d)
        return d

    def dist_leitner(self, user: User, card: Card) -> float:
        # time till scheduled date by Leitner
        t = user.leitner_scheduled_time.get(card.card_id, None)
        if t is None:
            return 0
        else:
            return max(0, (t - card.date).days)

    def dist_sm2(self, user: User, card: Card) -> float:
        # time till scheduled date by sm2
        t = user.sm2_scheduled_time.get(card.card_id, None)
        if t is None:
            return 0
        else:
            return max(0, (t - card.date).days)

    def score(self, user: User, cards: List[Card]) -> List[float]:
        return [{
            'qrep': self.dist_qrep(user, card),
            'skill': self.dist_skill(user, card),
            'time': self.dist_time(user, card),
            'category': self.dist_category(user, card),
            'leitner': self.dist_leitner(user, card),
            'sm2': self.dist_sm2(user, card),
        } for card in cards]

    def schedule(self, cards: List[dict]) -> Dict:
        if len(cards) == 0:
            return [], [], ''

        # a schedule request might contain that of many users
        # first group cards by user
        # card index for each user
        user_card_index_mapping = defaultdict(list)
        for i, c in enumerate(cards):
            user_card_index_mapping[c['user_id']].append(i)
        all_cards = self.get_cards(cards)

        users = [self.get_user(x) for x in user_card_index_mapping.keys()]
        # NOTE assuming single user here
        assert len(users) == 1

        # for user in users:  # not in use since we assume single user
        user = users[0]
        cards = [all_cards[i] for i in user_card_index_mapping[user.user_id]]

        scores = self.score(user, cards)
        scores_summed = [
            sum([self.params.__dict__[key] * value for key, value in ss.items()])
            for ss in scores
        ]
        order = np.argsort(scores_summed).tolist()
        # ranking = [order.index(i) for i, _ in enumerate(cards)]

        # create rationale
        index_selected = order[0]
        card_selected = cards[index_selected]
        topic_idx = np.argmax(card_selected.qrep)
        # scores before scaled of the selected card
        detail = scores[index_selected]
        detail.update({
            'sum': scores_summed[index_selected],
            'topic': self.topic_words[topic_idx],
        })

        rationale = '\n'.join([
            '{}: {}'.format(key, value) for key, value in detail.items()
        ])

        # add temporary history
        # ID and response will be completed by a call to `update`
        temp_history_id = json.dumps({'user_id': user.user_id,
                                      'card_id': card_selected.card_id})

        history = History(
            history_id=temp_history_id,
            user_id=user.user_id,
            card_id=card_selected.card_id,
            response='PLACEHOLDER',
            judgement='PLACEHOLDER',
            user_snapshot=user.to_snapshot(),
            scheduler_snapshot=json.dumps(self.params.__dict__),
            card_ids=json.dumps([x.card_id for x in cards]),
            scheduler_output=json.dumps({
                'order': order,
                'rationale': rationale}),
            date=datetime.now())
        self.db.add_history(history)
        return {
            'order': order,
            'rationale': rationale,
            'detail': detail
        }

    def update(self, cards: List[dict]):
        # where in the list are cards for each user
        user_card_index_mapping = defaultdict(list)
        responses, history_ids, dates = [], [], []
        for i, c in enumerate(cards):
            user_card_index_mapping[c['user_id']].append(i)
            responses.append(c['label'])
            history_ids.append(c['history_id'])
            dates.append(parse_date(c['date']))

        cards = self.get_cards(cards)

        for u, indices in user_card_index_mapping.items():
            user = self.get_user(u)
            for i in indices:
                card = cards[i]
                response = responses[i]
                history_id = history_ids[i]
                date = parse_date(dates[i])

                # update qrep
                user.qrep.append(card.qrep)
                if len(user.qrep) >= MAX_QREPS:
                    user.qrep.pop(0)

                # update skill
                if response == 'correct':
                    user.skill.append(card.skill)
                    if len(user.skill) >= MAX_QREPS:
                        user.skill.pop(0)

                user.category = card.category
                user.last_study_time[card.card_id] = date

                self.leitner_update(user, card, response)
                self.sm2_update(user, card, response)

                # find that temporary history entry and update
                temp_history_id = json.dumps({'user_id': user.user_id, 'card_id': card.card_id})
                history = self.db.get_history(temp_history_id)
                if history is not None:
                    history.__dict__.update({
                        'history_id': history_id,
                        'response': response,
                        'judgement': response,
                        'date': date
                    })
                    self.db.update_history(temp_history_id, history)
            self.db.update_user(user)

    def leitner_update(self, user: User, card: Card, response: str):
        # leitner boxes 1~10, None as placeholder since we don't have box 0
        days = [None, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 9999]
        increment_days = {i: x for i, x in enumerate(days)}

        # boxes: 1 ~ 10
        cur_box = user.leitner_box.get(card.card_id, None)
        if cur_box is None:
            cur_box = 1
        new_box = cur_box + (1 if response == 'correct' else -1)
        new_box = max(min(new_box, 10), 1)
        user.leitner_box[card.card_id] = new_box
        interval = timedelta(days=increment_days[new_box])
        date_studied = user.last_study_time[card.card_id]
        user.leitner_scheduled_time[card.card_id] = date_studied + interval

    def get_quality_from_response(self, response: str) -> int:
        return 4 if response == 'correct' else 1

    def sm2_update(self, user: User, card: Card, response: str):
        e_f = user.sm2_efactor.get(card.card_id, 2.5)
        inv = user.sm2_interval.get(card.card_id, 1)
        rep = user.sm2_repetition.get(card.card_id, 0) + 1

        q = self.get_quality_from_response(response)
        e_f = max(1.3, e_f + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

        if response != 'correct':
            inv = 0
            rep = 0
        else:
            if rep == 1:
                inv = 1
            elif rep == 2:
                inv = 6
            else:
                inv = inv * e_f

        user.sm2_repetition[card.card_id] = rep
        user.sm2_efactor[card.card_id] = e_f
        user.sm2_interval[card.card_id] = inv
        date_studied = user.last_study_time[card.card_id]
        user.sm2_scheduled_time[card.card_id] = date_studied + timedelta(days=inv)
