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
from typing import List, Dict

import gensim
import en_core_web_lg

# from whoosh.qparser import QueryParser


from db import SchedulerDB
from util import ScheduleRequest, Params, Card, User, History
from retention import RetentionModel
from build_lda import process

nlp = en_core_web_lg.load()
nlp.add_pipe(process, name='process', last=True)

logger = logging.getLogger('scheduler')


class MovingAvgScheduler:

    def __init__(self, params=None, db_filename='db.sqlite'):
        if params is None:
            params = Params()
        self.params = params
        self.db_filename = db_filename
        self.db = SchedulerDB(db_filename)
        self.retention_model = RetentionModel()

        '''
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
        from whoosh.index import open_dir
        self.ix = open_dir(self.params.whoosh_index)
        '''

        # LDA gensim
        # self.lda = gensim.models.LdaModel.load(os.path.join(params.lda_dir, 'lda'))
        self.lda = gensim.models.ldamulticore.LdaMulticore.load(os.path.join(params.lda_dir, 'lda'))
        self.vocab = gensim.corpora.Dictionary.load_from_text(os.path.join(params.lda_dir, 'vocab.txt'))
        with open(os.path.join(params.lda_dir, 'topic_words.txt'), 'r') as f:
            self.topic_words = [l.strip() for l in f.readlines()]
        with open(os.path.join(params.lda_dir, 'args.json'), 'r') as f:
            self.params.n_topics = json.load(f)['n_topics']
        logger.info(self.topic_words)
        # build default estimate for users
        self.avg_user_skill_estimate = self.estimate_avg()

    def estimate_avg(self) -> np.ndarray:
        # estimate the average acccuracy for each component
        # use for initializing user estimate

        estimate_file_dir = os.path.join(
            self.params.lda_dir, 'diagnostic_avg_estimate.txt')
        if os.path.exists(estimate_file_dir):
            logger.info('load user skill estimate')
            with open(estimate_file_dir) as f:
                return np.array([float(x) for x in f.readlines()])

        with open('data/diagnostic_questions.pkl', 'rb') as f:
            cards = pickle.load(f)

        logger.info('estimate average user skill')
        cards = [Card(
            card_id=c['question_id'],
            text=c['text'],
            answer=c['answer'],
            category=c['category'],
            qrep=None,
            skill=None) for c in cards]

        self.embed(cards)
        qreps = [c.qrep for c in cards]
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

    def reset(self, user_id=None):
        # delete users and history from db
        self.db.delete_user(user_id=user_id)
        self.db.delete_history(user_id=user_id)

    def build_whoosh(self):
        from whoosh.fields import Schema, ID, TEXT
        from whoosh.index import create_in

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

    def embed(self, cards: List[Card]):
        texts = [c.text for c in cards]
        texts = (self.vocab.doc2bow(x) for x in nlp.pipe(texts))
        # need to set minimum_probability to a negative value
        # to prevent gensim output skipping topics
        doc_topic_dists = self.lda.get_document_topics(texts, minimum_probability=-1)
        for card, dist in zip(cards, doc_topic_dists):
            # dist is something like [(d_i, i)]
            card.qrep = np.asarray([d_i for i, d_i in dist])

    def get_card(self, request: ScheduleRequest) -> Card:
        '''get card from db, insert if new'''
        # retrieve from db if exists
        card = self.db.get_card(request.question_id)
        if card is not None:
            return card

        card = Card(
            card_id=request.question_id,
            text=request.text,
            answer=request.answer,
            category=request.category,
            qrep=None,
            skill=None,
        )

        self.embed([card])
        card.skill = np.zeros_like(card.qrep)
        card.skill[np.argmax(card.qrep)] = 1
        card.skill *= self.predict_one(card)

        self.db.add_card(card)
        return card

    def get_cards(self, requests: List[ScheduleRequest]) -> List[Card]:
        new_cards, cards = [], []
        for i, r in enumerate(requests):
            card = self.db.get_card(r.question_id)
            if card is None:
                card = Card(
                    card_id=r.question_id,
                    text=r.text,
                    answer=r.answer,
                    category=r.category,
                    qrep=None,  # placeholder
                    skill=None  # placeholder
                )
                new_cards.append(card)
            cards.append(card)

        if len(new_cards) == 0:
            return cards

        logger.debug('embed cards ' + str(len(new_cards)))
        self.embed(new_cards)
        probs = self.predict(new_cards)
        for i, card in enumerate(new_cards):
            card.skill = np.zeros_like(card.qrep)
            card.skill[np.argmax(card.qrep)] = 1
            card.skill *= probs[i]
        self.db.add_cards(new_cards)
        return cards

    def get_user(self, user_id: str) -> User:
        '''get user from DB, insert if new'''
        # retrieve from db if exists
        user = self.db.get_user(user_id)
        if user is not None:
            return user

        # create new user and insert to db
        k = self.params.n_topics
        qrep = np.array([1 / k for _ in range(k)])
        new_user = User(
            user_id=user_id,
            category='HISTORY',  # TODO don't hard code
            qrep=[qrep],
            skill=[self.avg_user_skill_estimate]
        )
        self.db.add_user(new_user)
        return new_user

    '''
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
    '''

    def predict_one(self, user: User, card: Card) -> float:
        # # 1. find same or similar card in records
        # cards, scores = self.retrieve(card)
        # if len(cards) > 0:
        #     return np.dot([x['prob'] for x in cards], scores)
        # # TODO 2. use model to predict
        return self.retention_model.predict(user, [card])[0]

    def predict(self, user: User, cards: List[Card]) -> List[float]:
        return self.retention_model.predict(user, cards)

    def set_params(self, params: Params):
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
            weight *= self.params.decay_skill
        skill = np.sum(skills, axis=0) / np.sum(weights)
        # skill is a (vector of) probability
        skill = np.clip(skill, a_min=0.0, a_max=1.0)
        topic_idx = np.argmax(card.qrep)
        d = card.skill[topic_idx] - skill[topic_idx]
        # penalize easier questions by ten
        d *= 1 if d <= 0 else 10
        return abs(d)

    def dist_time(self, user: User, card: Card, date: datetime) -> float:
        # cool down question for 10 minutes
        last_study_date = user.last_study_date.get(card.card_id, None)
        if last_study_date is None:
            return 0
        else:
            current_date = time.mktime(date.timetuple())
            last_study_date = time.mktime(last_study_date.timetuple())
            delta_minutes = max(float(current_date - last_study_date) / 60, 0)
            return max(self.params.cool_down_time - delta_minutes, 0)

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
        return cosine_distance(qrep, card.qrep)

    def dist_leitner(self, user: User, card: Card, date: datetime) -> float:
        # days till scheduled date by Leitner
        scheduled_date = user.leitner_scheduled_date.get(card.card_id, None)
        if scheduled_date is None:
            return 0
        else:
            # distance in hours
            return max(0, (scheduled_date - date).seconds / (60 * 60))

    def dist_sm2(self, user: User, card: Card, date: datetime) -> float:
        # days till scheduled date by sm2
        scheduled_date = user.sm2_scheduled_date.get(card.card_id, None)
        if scheduled_date is None:
            return 0
        else:
            # distance in hours
            return max(0, (scheduled_date - date).seconds / (60 * 60))

    def score(self, user: User, cards: List[Card], date: datetime) -> List[float]:
        return [{
            'qrep': self.dist_qrep(user, card),
            'skill': self.dist_skill(user, card),
            'category': self.dist_category(user, card),
            'time': self.dist_time(user, card, date),
            'leitner': self.dist_leitner(user, card, date),
            'sm2': self.dist_sm2(user, card, date),
        } for card in cards]

    def schedule(self, requests: List[ScheduleRequest], date: datetime) -> Dict:
        if len(requests) == 0:
            # TODO raise exception and return something more meaningful
            return [], [], ''

        t0 = datetime.now()
        user_to_requests = defaultdict(list)
        for i, request in enumerate(requests):
            user_to_requests[request.user_id].append(i)

        cards = self.get_cards(requests)

        # for u, indices in user_to_requests.items():
        assert len(user_to_requests) == 1
        user, indices = list(user_to_requests.items())[0]
        user = self.get_user(user)

        # cards = [cards[i] for i in user_to_requests[user.user_id]]

        scores = self.score(user, cards, date)
        scores_summed = [
            sum([self.params.__dict__[key] * value for key, value in ss.items()])
            for ss in scores
        ]
        order = np.argsort(scores_summed).tolist()

        # create rationale
        index_selected = order[0]
        card_selected = cards[index_selected]
        topic_idx = np.argmax(card_selected.qrep)
        detail = scores[index_selected]
        detail.update({
            'sum': scores_summed[index_selected],
            'topic': self.topic_words[topic_idx],
            'current date': date,
            'last study date': user.last_study_date.get(card_selected.card_id, '-')
        })

        rationale = ''
        for key, value in detail.items():
            if isinstance(value, float):
                rationale += '{}: {:.4f}\n'.format(key, value)
            else:
                rationale += '{}: {}\n'.format(key, value)

        logger.debug(card_selected.answer)

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
            user_snapshot=json.dumps(user.pack()),
            scheduler_snapshot=json.dumps(self.params.__dict__),
            card_ids=json.dumps([x.card_id for x in cards]),
            scheduler_output=json.dumps({'order': order, 'rationale': rationale}),
            date=date)
        self.db.add_history(history)

        t1 = datetime.now()
        logger.debug('schedule ' + str(t1 - t0))

        return {
            'order': order,
            'rationale': rationale,
            'detail': detail
        }

    def update(self, requests: List[ScheduleRequest], date: datetime) -> Dict:
        user_to_requests = defaultdict(list)
        for i, request in enumerate(requests):
            user_to_requests[request.user_id].append(i)

        cards = self.get_cards(requests)

        # for u, indices in user_to_requests.items():
        assert len(user_to_requests) == 1
        user, indices = list(user_to_requests.items())[0]
        user = self.get_user(user)

        # for i in indices:
        assert len(indices) == 1
        request = requests[indices[0]]
        card = cards[indices[0]]
        response = request.label == 'correct'

        detail = {
            # 'user_id': user.user_id,
            # 'card_id': card.card_id,
            # 'answer': card.answer,
            'response': request.label,
            # 'last_study_date': user.last_study_date.get(card.card_id, '-'),
            # 'current_date': date,
        }

        tmp = {
            'old ltn box': user.leitner_box.get(card.card_id, '-'),
            'old ltn dat': user.leitner_scheduled_date.get(card.card_id, '-'),
            'old sm2 rep': user.sm2_repetition.get(card.card_id, '-'),
            'old sm2 inv': user.sm2_interval.get(card.card_id, '-'),
            'old sm2 e_f': user.sm2_efactor.get(card.card_id, '-'),
            'old sm2 dat': user.sm2_scheduled_date.get(card.card_id, '-')
        }

        # update qrep
        user.qrep.append(card.qrep)
        if len(user.qrep) >= self.params.max_qreps:
            user.qrep.pop(0)

        # update skill
        if response:
            user.skill.append(card.skill)
            if len(user.skill) >= self.params.max_qreps:
                user.skill.pop(0)

        user.category = card.category
        user.last_study_date[card.card_id] = date

        # update retention features
        user.results.append(response)
        card.results.append(response)
        if card.card_id not in user.count_correct_before:
            user.count_correct_before[card.card_id] = 0
        if card.card_id not in user.count_wrong_before:
            user.count_wrong_before[card.card_id] = 0
        if response:
            user.count_correct_before[card.card_id] += 1
        else:
            user.count_wrong_before[card.card_id] += 1

        self.leitner_update(user, card, request.label)
        self.sm2_update(user, card, request.label)

        detail.update({
            'old ltn box': tmp['old ltn box'],
            'new ltn box': user.leitner_box.get(card.card_id, '-'),
            'old ltn dat': tmp['old ltn dat'],
            'new ltn dat': user.leitner_scheduled_date.get(card.card_id, '-'),
            'old sm2 rep': tmp['old sm2 rep'],
            'new sm2 rep': user.sm2_repetition.get(card.card_id, '-'),
            'old sm2 inv': tmp['old sm2 inv'],
            'new sm2 inv': user.sm2_interval.get(card.card_id, '-'),
            'old sm2 e_f': tmp['old sm2 e_f'],
            'new sm2 e_f': user.sm2_efactor.get(card.card_id, '-'),
            'old sm2 dat': tmp['old sm2 dat'],
            'new sm2 dat': user.sm2_scheduled_date.get(card.card_id, '-')
        })

        # find that temporary history entry and update
        temp_history_id = json.dumps({'user_id': user.user_id, 'card_id': card.card_id})
        history = self.db.get_history(temp_history_id)
        if history is not None:
            history.__dict__.update({
                'history_id': request.history_id,
                'response': request.label,
                'judgement': request.label,
                'date': date
            })
            self.db.update_history(temp_history_id, history)
        # TODO else add_history

        self.db.update_user(user)
        self.db.update_card(card)

        return detail

    def leitner_update(self, user: User, card: Card, response: str):
        # leitner boxes 1~10
        # days[0] = None as placeholder since we don't have box 0
        # days[9] and days[10] = 9999 to make it never repeat
        days = [None, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 9999, 9999]
        increment_days = {i: x for i, x in enumerate(days)}

        # boxes: 1 ~ 10
        cur_box = user.leitner_box.get(card.card_id, None)
        if cur_box is None:
            cur_box = 1
        new_box = cur_box + (1 if response == 'correct' else -1)
        new_box = max(min(new_box, 10), 1)
        user.leitner_box[card.card_id] = new_box
        interval = timedelta(days=increment_days[new_box])
        date_studied = user.last_study_date[card.card_id]
        user.leitner_scheduled_date[card.card_id] = date_studied + interval

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
        date_studied = user.last_study_date[card.card_id]
        user.sm2_scheduled_date[card.card_id] = date_studied + timedelta(days=inv)
