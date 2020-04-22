#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import copy
import pickle
import gensim
import logging
import numpy as np
import pandas as pd
import en_core_web_lg
from tqdm import tqdm
from collections import defaultdict
from datetime import datetime, timedelta
from typing import List, Dict
# from whoosh.qparser import QueryParser

from plotnine import ggplot, aes, geom_bar, coord_flip
from pandas.api.types import CategoricalDtype

from karl.db import SchedulerDB
from karl.lda import process_question
from karl.util import ScheduleRequest, Params, Card, User, History, theme_fs
from karl.baseline import RetentionModel

nlp = en_core_web_lg.load()
nlp.add_pipe(process_question, name='process', last=True)

logger = logging.getLogger('scheduler')

# def plot_polar(qrep, filename=None, alpha=0.2, fill='blue'):
#     qrep = qrep.tolist()
#     n_topics = len(qrep)
#
#     plt.figure(figsize=(3, 3))
#     ax = plt.subplot(111, projection='polar')
#
#     theta = np.linspace(0, 2 * np.pi, n_topics)
#     # topics = ['topic_%d' % i for i in range(n_topics)]
#     # lines, labels = plt.thetagrids(range(0, 360, int(360 / n_topics)), (topics))
#
#     ax.plot(theta, qrep)
#     ax.fill(theta, qrep, fill=fill, alpha=alpha / 2)
#
#     # remove ticks
#     ax.set_rticks([])
#
#     # Add legend and title for the plot
#     # ax.legend(labels=('Card', 'User', 'Next'), loc=1)
#     # ax.set_title("Question representation")
#
#     # Dsiplay the plot on the screen
#     # plt.show()
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#     plt.close()


class MovingAvgScheduler:

    def __init__(self, params=None, db_filename='db.sqlite'):
        if params is None:
            params = Params()
        self.params = params
        self.db_filename = db_filename
        self.db = SchedulerDB(db_filename)
        self.retention_model = RetentionModel()

        # logger.info('loading question and records...')
        # with open('data/jeopardy_310326_question_player_pairs_20190612.pkl', 'rb') as f:
        #     records_df = pickle.load(f)
        # with open('data/jeopardy_358974_questions_20190612.pkl', 'rb') as f:
        #     questions_df = pickle.load(f)
        # self.karl_to_question_id = questions_df.to_dict()['question_id']

        # # augment question_df with record stats
        # logger.info('merging dfs...')
        # df = questions_df.set_index('question_id').join(records_df.set_index('question_id'))
        # df = df[df['correct'].notna()]
        # df['correct'] = df['correct'].apply(lambda x: 1 if x == 1 else 0)
        # df_grouped = df.reset_index()[['question_id', 'correct']].groupby('question_id')
        # dict_correct_mean = df_grouped.mean().to_dict()['correct']
        # dict_records_cnt = df_grouped.count().to_dict()['correct']
        # questions_df['prob'] = questions_df['question_id'].apply(lambda x: dict_correct_mean.get(x, None))
        # questions_df['count'] = questions_df['question_id'].apply(lambda x: dict_records_cnt.get(x, None))
        # self.questions_df = questions_df
        # self.question_id_set = set(self.questions_df['question_id'])

        # # setup whoosh for text search
        # if not os.path.exists(self.params.whoosh_index):
        #     logger.info('building whoosh...')
        #     self.build_whoosh()
        # from whoosh.index import open_dir
        # self.ix = open_dir(self.params.whoosh_index)

        # LDA gensim
        # self.lda_model = gensim.models.LdaModel.load(os.path.join(params.lda_dir, 'lda'))
        self.lda_model = gensim.models.ldamulticore.LdaMulticore.load(os.path.join(params.lda_dir, 'lda'))
        self.vocab = gensim.corpora.Dictionary.load_from_text(os.path.join(params.lda_dir, 'vocab.txt'))
        with open(os.path.join(params.lda_dir, 'topic_words.txt'), 'r') as f:
            self.topic_words = [l.strip() for l in f.readlines()]
        with open(os.path.join(params.lda_dir, 'args.json'), 'r') as f:
            self.params.n_topics = json.load(f)['n_topics']
        logger.info(self.topic_words)
        # build default estimate for users
        self.avg_user_skill_estimate = self.estimate_avg()

        # cache of precomputed results
        # user.user_id
        # | user
        # | cards
        # | card
        # | date
        # | 'correct'
        # | | user
        # | | order
        # | | scores
        # | | rationale
        # | | cards_info
        # | | (plot)
        # | 'wrong'
        # | | user
        # | | order
        # | | scores
        # | | rationale
        # | | cards_info
        # | | (plot)
        # | response: actual response, filled in by update
        self.precomputed_results = dict()

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
        doc_topic_dists = self.lda_model.get_document_topics(texts, minimum_probability=-1)
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

    # def retrieve(self, card: dict) -> Tuple[List[dict], List[float]]:
    #     record_id = self.karl_to_question_id[int(card['question_id'])]

    #     # 1. try to find in records with gameid-catnum-level
    #     if record_id in self.question_id_set:
    #         hits = self.questions_df[self.questions_df.question_id == record_id]
    #         if len(hits) > 0:
    #             cards = [card.to_dict() for idx, card in hits.iterrows()]
    #             return cards, [1 / len(hits) for _ in range(len(hits))]
    #     else:
    #         # return better default value without text search
    #         return 0.5

    #     # 2. do text search
    #     with self.ix.searcher() as searcher:
    #         query = QueryParser("text", self.ix.schema).parse(card['text'])
    #         hits = searcher.search(query)
    #         hits = [x for x in hits if x['answer'] == card['answer']]
    #         if len(hits) > 0:
    #             scores = [x.score for x in hits]
    #             ssum = np.sum(scores)
    #             scores = [x / ssum for x in scores]
    #             cards = [self.questions_df[self.questions_df['question_id'] == x['question_id']].iloc[0] for x in hits]
    #             return cards, scores

    #     # 3. not found
    #     return [], []

    def predict_one(self, card: Card) -> float:
        # # 1. find same or similar card in records
        # cards, scores = self.retrieve(card)
        # if len(cards) > 0:
        #     return np.dot([x['prob'] for x in cards], scores)
        # # 2. use model to predict
        # return self.retention_model.predict_one(user, card)
        return 0.5

    def predict(self, cards: List[Card]) -> List[float]:
        # return self.retention_model.predict(user, cards)
        return [self.predict_one(card) for card in cards]

    def set_params(self, params: Params):
        self.params.__dict__.update(params)

    def dist_category(self, user: User, card: Card) -> float:
        if user.category is None or card.category is None:
            return 0
        return float(card.category.lower() != user.category.lower())

    def dist_skill(self, user: User, card: Card) -> float:
        return 0
        # # measures difficulty difference
        # weight = 1
        # skills, weights = [], []
        # for q in user.skill:
        #     skills.append(weight * q)
        #     weights.append(weight)
        #     weight *= self.params.decay_skill
        # skill = np.sum(skills, axis=0) / np.sum(weights)
        # # skill is a (vector of) probability
        # skill = np.clip(skill, a_min=0.0, a_max=1.0)
        # topic_idx = np.argmax(card.qrep)
        # d = card.skill[topic_idx] - skill[topic_idx]
        # # penalize easier questions by ten
        # d *= 1 if d <= 0 else 10  # TODO add to param
        # return abs(d)

    def dist_recall_batch(self, user: User, cards: List[Card]) -> float:
        # return [0 for _ in cards]
        return (1 - self.retention_model.predict(user, cards)).tolist()

    def dist_cool_down(self, user: User, card: Card, date: datetime) -> float:
        prev_date, prev_response = user.previous_study.get(card.card_id, (None, None))
        if prev_date is None:
            return 0
        else:
            current_date = time.mktime(date.timetuple())
            prev_date = time.mktime(prev_date.timetuple())
            delta_minutes = max(float(current_date - prev_date) / 60, 0)
            # cool down is never negative
            if prev_response == 'correct':
                return max(self.params.cool_down_time_correct - delta_minutes, 0)
            else:
                return max(self.params.cool_down_time_wrong - delta_minutes, 0)

    def get_average_qrep(self, qreps):
        weight = 1
        qrep, weights = [], []
        for q in qreps:
            qrep.append(weight * q)
            weights.append(weight)
            weight *= self.params.decay_qrep
        qrep = np.sum(qrep, axis=0) / np.sum(weights)
        return qrep

    def dist_qrep(self, user: User, card: Card) -> float:
        # measures topical similarity
        def cosine_distance(a, b):
            return 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        qrep = self.get_average_qrep(user.qrep)
        return cosine_distance(qrep, card.qrep)

    def dist_leitner(self, user: User, card: Card, date: datetime) -> float:
        # hours till scheduled date by Leitner
        scheduled_date = user.leitner_scheduled_date.get(card.card_id, None)
        if scheduled_date is None:
            return 0
        else:
            # NOTE distance in hours, can be negative
            return (scheduled_date - date).total_seconds() / (60 * 60)

    def dist_sm2(self, user: User, card: Card, date: datetime) -> float:
        # hours till scheduled date by sm2
        scheduled_date = user.sm2_scheduled_date.get(card.card_id, None)
        if scheduled_date is None:
            return 0
        else:
            # NOTE distance in hours, can be negative
            return (scheduled_date - date).total_seconds() / (60 * 60)

    def score(self, user: User, cards: List[Card], date: datetime) -> List[float]:
        recall_scores = self.dist_recall_batch(user, cards)
        return [{
            'qrep': self.dist_qrep(user, card),
            # 'skill': self.dist_skill(user, card),
            'recall': recall_scores[i],
            'category': self.dist_category(user, card),
            'cool_down': self.dist_cool_down(user, card, date),
            'leitner': self.dist_leitner(user, card, date),
            'sm2': self.dist_sm2(user, card, date),
        } for i, card in enumerate(cards)]
    
    def schedule_and_predict(self,
                             requests: List[ScheduleRequest],
                             date: datetime,
                             plot: bool = True) -> Dict:
        # 0. (update) merge with reality
        # 1. schedule
        #    1.1 check if new cards got added
        #    1.2 if new cards, get scores for new cards, rank, return
        #    1.3 if not new cards, return cached results
        # 2. copy user, update with response=True, schedule again, cache results
        # 3. copy user, update with response=False, schedule again, cache results
        # cached results: User -> (cards, scores)

        if len(requests) == 0:
            return [], [], ''

        # mapping from user to scheduling requests
        # not used since we assume only one user
        user_to_requests = defaultdict(list)
        for i, request in enumerate(requests):
            user_to_requests[request.user_id].append(i)
        if len(user_to_requests) != 1:
            raise ValueError('Schedule only accpets 1 user. Received {}'.format(len(user_to_requests)))

        cards = self.get_cards(requests)
        user, indices = list(user_to_requests.items())[0]
        user = self.get_user(user)
        cards = [cards[i] for i in indices]

        if user.user_id not in self.precomputed_results:
            results = self.schedule(user, cards, date, add_history=True, plot=plot)
        elif not self.precomputed_results[user.user_id].get('response', None):
            # previous study did not get an update, throw away precomputed results
            results = self.schedule(user, cards, date, add_history=True, plot=plot)
        else:
            # precomputed_results[user.user_id]
            # | user
            # | cards
            # | date
            # | 'correct'
            # | | user
            # | | order
            # | | scores
            # | | rationale
            # | | cards_info
            # | | (plot)
            # | 'wrong'
            # | | user
            # | | order
            # | | scores
            # | | rationale
            # | | cards_info
            # | | (plot)
            # | response: actual response, filled in by update
            prev_results = self.precomputed_results[user.user_id]
            prev_cards = prev_results['cards']
            prev_card_ids = [c.card_id for c in prev_cards]
            # prev_cards -> cards
            prev_card_indices = [None] * len(prev_cards)

            # choose the one that matches reality
            prev_results_reality = prev_results[prev_results['response']]

            # new_cards -> cards
            new_cards, new_card_indices = [], []
            for i, c in enumerate(cards):
                if c.card_id in prev_card_ids:
                    prev_card_indices[prev_card_ids.index(c.card_id)] = i
                else:
                    new_cards.append(c)
                    new_card_indices.append(i)

            if len(new_cards) == 0:
                results = prev_results_reality
            else:
                new_results = self.schedule(user, new_cards, date, add_history=False, plot=plot)

                scores = [None] * len(cards)
                # new_cards -> cards
                for i, idx in enumerate(new_card_indices):
                    scores[idx] = new_results['scores'][i]
                # prev_cards -> cards
                for i, idx in enumerate(prev_card_indices):
                    scores[idx] = prev_results_reality['scores'][i]
        
                scores_summed = [s['sum'] for s in scores]
                order = np.argsort(scores_summed).tolist()

                # TODO get new rationale and cards_info
                results = {
                    'order': order,
                    'rationale': prev_results_reality['rationale'],
                    'cards_info': prev_results_reality['cards_info'],
                    'scores': scores,
                }

        card = cards[results['order'][0]]
        user_correct = copy.deepcopy(user)
        user_wrong = copy.deepcopy(user)
        # TODO update_with_response also changes card
        self.update_with_response(user_correct, card, date, 'correct')
        self.update_with_response(user_wrong, card, date, 'correct')
        results_correct = self.schedule(user_correct, cards, date, add_history=False, plot=plot)
        results_wrong = self.schedule(user_wrong, cards, date, add_history=False, plot=plot)
        # cached results: User -> (cards, scores)
        self.precomputed_results[user.user_id] = {
            'user': user,
            'cards': cards,
            'card_id': card.card_id,
            'date': date,
            'correct': {
                'user': user_correct,
                'order': results_correct['order'],
                'scores': results_correct['scores'],
                'rationale': results_correct['rationale'],
                'cards_info': results_correct['cards_info'],
            },
            'wrong': {
                'user': user_wrong,
                'order': results_wrong['order'],
                'scores': results_wrong['scores'],
                'rationale': results_wrong['rationale'],
                'cards_info': results_wrong['cards_info'],
            },
            'response': None,
        }

        return results
        
    def schedule(self,
                 user: User,
                 cards: List[Card],
                 date: datetime,
                 add_history: bool = True,
                 plot: bool = True) -> Dict:
        scores = self.score(user, cards, date)
        scores_summed = [
            sum([self.params.__dict__[key] * value for key, value in ss.items()])
            for ss in scores
        ]
        for i, ss in enumerate(scores):
            ss['sum'] = scores_summed[i]

        order = np.argsort(scores_summed).tolist()
        card = cards[order[0]]

        user_qrep = self.get_average_qrep(user.qrep)
        # plot_polar(user_qrep, '/fs/www-users/shifeng/temp/user.jpg', fill='green')
        # plot_polar(card.qrep, '/fs/www-users/shifeng/temp/card.jpg', fill='yellow')

        if plot:
            figname = '{}_{}_{}.jpg'.format(user.user_id, card.card_id, date.strftime('%Y-%m-%d-%H-%M'))
            local_filename = '/fs/www-users/shifeng/temp/' + figname
            remote_filename = 'http://users.umiacs.umd.edu/~shifeng/temp/' + figname
            self.plot_histogram(card.qrep, user_qrep, local_filename)

        # create rationale
        cards_info = [copy.deepcopy(c.__dict__) for c in cards]
        for i, card_info in enumerate(cards_info):
            card_info['scores'] = scores[i]
            prev_date, prev_response = user.previous_study.get(card_info['card_id'], ('-', '-'))
            card_info.update({
                'current date': date,
                'topic': self.topic_words[np.argmax(card_info['qrep'])],
                'prev_response': prev_response,
                'prev_date': prev_date
            })
            card_info.pop('qrep')
            card_info.pop('skill')
            card_info.pop('results')
        cards_info = [cards_info[i] for i in order[:3]]

        rr = """
             <style>
             table {
               border-collapse: collapse;
             }

             td, th {
               padding: 0.5rem;
               text-align: left;
             }

             tr:nth-child(even) {background-color: #f2f2f2;}
             </style>
             """
        # print('', file=detail_file)
        # print(' ' * 3, 'update details', file=detail_file)
        # for key, value in update_outputs.items():
        #     if isinstance(value, float):
        #         print(' ' * 7, '{: <16} : {:.4f}'.format(key, value), file=detail_file)
        #     elif isinstance(value, int) or isinstance(value, str):
        #         print(' ' * 7, '{: <16} : {}'.format(key, value), file=detail_file)

        rr += '<h2>Top ranked cards</h2>'
        row_template = '<tr><td><b>{}</b></td> <td>{}</td></tr>'
        row_template_3 = '<tr><td><b>{}</b></td> <td>{:.4f} x {:.2f}</td></tr>'
        for i in order[:3]:
            c = cards[i]
            prev_date, prev_response = user.previous_study.get(c.card_id, ('-', '-'))
            rr += '<table style="float: left;">'
            rr += row_template.format('card_id', c.card_id)
            rr += row_template.format('answer', c.answer)
            rr += row_template.format('category', c.category)
            rr += row_template.format('topic', self.topic_words[np.argmax(c.qrep)])
            rr += row_template.format('prev_date', prev_date)
            rr += row_template.format('prev_response', prev_response)
            for k, v in scores[i].items():
                rr += row_template_3.format(k, v, self.params.__dict__.get(k, 0))
            rr += '<tr><td><b>{}</b></td> <td>{:.4f}</td></tr>'.format('sum', scores_summed[i])
            rr += row_template.format('ltn box', user.leitner_box.get(c.card_id, '-'))
            rr += row_template.format('ltn dat', user.leitner_scheduled_date.get(c.card_id, '-'))
            rr += row_template.format('sm2 rep', user.sm2_repetition.get(c.card_id, '-'))
            rr += row_template.format('sm2 inv', user.sm2_interval.get(c.card_id, '-'))
            rr += row_template.format('sm2 e_f', user.sm2_efactor.get(c.card_id, '-'))
            rr += row_template.format('sm2 dat', user.sm2_scheduled_date.get(c.card_id, '-'))
            rr += '</table>'

        output_dict = {
            'order': order,
            'rationale': rr,
            'cards_info': cards_info,
            'scores': scores,
        }

        if plot:
            rr += "</br>"
            rr += "</br>"
            rr += "<img src={}>".format(remote_filename)
            # rr += "<img src='http://users.umiacs.umd.edu/~shifeng/temp/card.jpg'>"
            # rr += "<img src='http://users.umiacs.umd.edu/~shifeng/temp/user.jpg'>"
            output_dict['plot'] = remote_filename

        if add_history:
            # add temporary history
            # ID and response will be completed by a call to `update`
            temp_history_id = json.dumps({'user_id': user.user_id, 'card_id': card.card_id})
            history = History(
                history_id=temp_history_id,
                user_id=user.user_id,
                card_id=card.card_id,
                response='PLACEHOLDER',
                judgement='PLACEHOLDER',
                user_snapshot=json.dumps(user.pack()),
                scheduler_snapshot=json.dumps(self.params.__dict__),
                card_ids=json.dumps([x.card_id for x in cards]),
                scheduler_output=json.dumps({'order': order, 'rationale': rr}),
                date=date)

            self.db.add_history(history)
            output_dict['temp_history_id'] = temp_history_id

        return output_dict

    def update_with_response(self, user: User, card: Card, date: datetime, response: str):
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
        user.previous_study[card.card_id] = (date, response)

        # update retention features
        card.results.append(response == 'correct')
        user.results.append(response == 'correct')
        if card.card_id not in user.count_correct_before:
            user.count_correct_before[card.card_id] = 0
        if card.card_id not in user.count_wrong_before:
            user.count_wrong_before[card.card_id] = 0
        if response:
            user.count_correct_before[card.card_id] += 1
        else:
            user.count_wrong_before[card.card_id] += 1

        self.leitner_update(user, card, response)
        self.sm2_update(user, card, response)

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

        old_user = copy.deepcopy(user)
        results = self.precomputed_results.get(user.user_id, None)
        if results is None:
            self.update_with_response(user, card, date, request.label)
        elif results['card_id'] != card.card_id:
            self.update_with_response(user, card, date, request.label)
            self.precomputed_results.pop(user.user_id)
        else:
            results['response'] = request.label
            user = results[request.label]['user']

        detail = {
            'response': request.label,
            'old ltn box': old_user.leitner_box.get(card.card_id, '-'),
            'new ltn box': user.leitner_box.get(card.card_id, '-'),
            'old ltn dat': str(old_user.leitner_scheduled_date.get(card.card_id, '-')),
            'new ltn dat': str(user.leitner_scheduled_date.get(card.card_id, '-')),
            'old sm2 rep': old_user.sm2_repetition.get(card.card_id, '-'),
            'new sm2 rep': user.sm2_repetition.get(card.card_id, '-'),
            'old sm2 inv': old_user.sm2_interval.get(card.card_id, '-'),
            'new sm2 inv': user.sm2_interval.get(card.card_id, '-'),
            'old sm2 e_f': old_user.sm2_efactor.get(card.card_id, '-'),
            'new sm2 e_f': user.sm2_efactor.get(card.card_id, '-'),
            'old sm2 dat': str(old_user.sm2_scheduled_date.get(card.card_id, '-')),
            'new sm2 dat': str(user.sm2_scheduled_date.get(card.card_id, '-')),
        }

        # find that temporary history entry and update
        # TODO with precompute enabled, we no longer write temporary history entry
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
        else:
            history = History(
                history_id=request.history_id,
                user_id=request.user_id,
                card_id=request.question_id,
                response=request.label,
                judgement=request.label,
                user_snapshot=json.dumps(user.pack()),
                scheduler_snapshot=json.dumps(self.params.__dict__),
                card_ids=json.dumps([x.card_id for x in cards]),
                scheduler_output='',
                date=date)
            self.db.add_history(history)

        self.db.update_user(user)
        self.db.update_card(card)

        print(' ' * 3, '{: <16} : {}'.format('card_id', card.card_id))
        print(' ' * 3, '{: <16} : {}'.format('answer', card.answer))
        for key, value in detail.items():
            if isinstance(value, float):
                print(' ' * 3, '{: <16} : {:.4f}'.format(key, value))
            elif isinstance(value, int) or isinstance(value, str):
                print(' ' * 3, '{: <16} : {}'.format(key, value))

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
        prev_date, prev_response = user.previous_study[card.card_id]
        user.leitner_scheduled_date[card.card_id] = prev_date + interval

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
        prev_date, prev_response = user.previous_study[card.card_id]
        user.sm2_scheduled_date[card.card_id] = prev_date + timedelta(days=inv)

    def plot_histogram(self, card_qrep, user_qrep, filename):
        max_qrep = np.max((card_qrep, user_qrep), axis=0)
        top_topics = np.argsort(-max_qrep)[:10]
        card_qrep = np.array(card_qrep)[top_topics].tolist()
        user_qrep = np.array(user_qrep)[top_topics].tolist()
        top_topic_words = [self.topic_words[i] for i in top_topics]
        topic_type = CategoricalDtype(categories=top_topic_words, ordered=True)
        df = pd.DataFrame({
            'topics': top_topic_words * 2,
            'weight': card_qrep + user_qrep,
            'label': ['card' for _ in top_topics] + ['user' for _ in top_topics]
        })
        df['topics'] = df['topics'].astype(str).astype(topic_type)

        p = (
            ggplot(df)
            + geom_bar(
                aes(x='topics', y='weight', fill='label'),
                stat='identity',
                position='identity',
                alpha=0.3,
            )
            + coord_flip()
            + theme_fs()
            # + theme(axis_text_x=(element_text(rotation=90)))
        )
        p.save(filename)
