#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import copy
import pickle
import gensim
import logging
import threading
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
from karl.util import ScheduleRequest, Params, Fact, User, History
from karl.util import leitner_params, sm2_params
from karl.util import theme_fs
from karl.retention.baseline import RetentionModel

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
#     # ax.legend(labels=('Fact', 'User', 'Next'), loc=1)
#     # ax.set_title("Question representation")
#
#     # Dsiplay the plot on the screen
#     # plt.show()
#     plt.savefig(filename, bbox_inches='tight', pad_inches=0)
#     plt.close()


class MovingAvgScheduler:

    def __init__(
            self,
            db_filename='db.sqlite',
            precompute=True,
            lda_dir='checkpoints/gensim_quizbowl_10_1585102364.5221019',
            whoosh_index='whoosh_index',
    ) -> None:
        """
        :param db_filename: location of database store.
        :param precompute: on-off switch of precompute optimization.
        :param lda_dir: gensim LDA model directory. 
        :param whoosh_index: whoosh index for text matching.
        """
        self.db_filename = db_filename
        self.precompute = precompute
        self.lda_dir = lda_dir
        self.whoosh_index = whoosh_index

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
        # if not os.path.exists(self.whoosh_index):
        #     logger.info('building whoosh...')
        #     self.build_whoosh()
        # from whoosh.index import open_dir
        # self.ix = open_dir(self.whoosh_index)

        # LDA gensim
        # self.lda_model = gensim.models.LdaModel.load(os.path.join(params.lda_dir, 'lda'))
        self.lda_model = gensim.models.ldamulticore.LdaMulticore.load(os.path.join(lda_dir, 'lda'))
        self.vocab = gensim.corpora.Dictionary.load_from_text(os.path.join(lda_dir, 'vocab.txt'))
        with open(os.path.join(lda_dir, 'topic_words.txt'), 'r') as f:
            self.topic_words = [l.strip() for l in f.readlines()]
        with open(os.path.join(lda_dir, 'args.json'), 'r') as f:
            self.n_topics = json.load(f)['n_topics']
        logger.info(self.topic_words)
        # build default estimate for users
        self.avg_user_skill_estimate = self.estimate_avg()

        # True
        # |user.user_id
        # | | user  # after branching update
        # | | fact  # previously selected fact, after branching update
        # | | facts # previously scored facts, after branching update
        # | | order
        # | | scores
        # | | rationale
        # | | facts_info
        # | | (plot)
        # False
        # |user.user_id
        # | | user  # after branching update
        # | | fact  # previously selected fact, after branching update
        # | | facts # previously scored facts, after branching update
        # | | order
        # | | scores
        # | | rationale
        # | | facts_info
        # | | (plot)
        self.precompute_future = {'correct': {}, 'wrong': {}}
        self.precompute_commit = dict()

    def estimate_avg(self) -> np.ndarray:
        """
        Estimate the average user acccuracy in each topic.
        This estimate is used as the initial user skill estimate.

        :return: a numpy array of size `n_topics` of average skill estimate of each topic.
        """
        estimate_file_dir = os.path.join(
            self.lda_dir, 'diagnostic_avg_estimate.txt')
        if os.path.exists(estimate_file_dir):
            logger.info('load user skill estimate')
            with open(estimate_file_dir) as f:
                return np.array([float(x) for x in f.readlines()])

        with open('data/diagnostic_questions.pkl', 'rb') as f:
            facts = pickle.load(f)

        logger.info('estimate average user skill')
        facts = [Fact(
            fact_id=c['fact_id'],
            text=c['text'],
            answer=c['answer'],
            category=c['category'],
            qrep=None,
            skill=None) for c in facts]

        self.embed(facts)
        qreps = [c.qrep for c in facts]
        estimates = [[] for _ in range(self.n_topics)]
        for fact, qrep in zip(facts, qreps):
            topic_idx = np.argmax(qrep)
            prob = self.predict_one(fact)
            estimates[topic_idx].append(prob)
        estimates = [np.mean(x) for x in estimates]
        with open(estimate_file_dir, 'w') as f:
            for e in estimates:
                f.write(str(e) + '\n')
        return np.array(estimates)

    def reset_user(self, user_id=None) -> None:
        """
        Delete a specific user (when `user_id` is provided) or all users (when `user_id` is `None`)
        and the corresponding history entries from database. This resets the study progress.

        :param user_id: the `user_id` of the user to be reset. Resets all users if `None`.
        """
        self.db.delete_user(user_id=user_id)
        self.db.delete_history(user_id=user_id)
        if user_id is not None:
            if user_id in self.precompute_future['correct']:
                self.precompute_future['correct'].pop(user_id)
            if user_id in self.precompute_future['wrong']:
                self.precompute_future['wrong'].pop(user_id)
            if user_id in self.precompute_commit:
                self.precompute_commit.pop(user_id)

    def reset_fact(self, fact_id=None) -> None:
        """
        Delete a specific fact (if `fact_id` is provided) or all facts (if `fact_id` is None) from
        database. This removed the cached embedding and skill estimate of the fact(s).

        :param fact_id: the `fact_id` of the fact to be reset. Resets all facts if `None`.
        """
        self.db.delete_fact(fact_id=fact_id)

    def build_whoosh(self) -> None:
        """
        Construct a whoosh index for text matching.
        The whoosh index is useful for finding similar facts with user records.
        """
        # NOTE This is not in use at the moment
        from whoosh.fields import Schema, ID, TEXT
        from whoosh.index import create_in

        if not os.path.exists(self.whoosh_index):
            os.mkdir(self.whoosh_index)
        schema = Schema(
            question_id=ID(stored=True),
            text=TEXT(stored=True),
            answer=TEXT(stored=True)
        )
        ix = create_in(self.whoosh_index, schema)
        writer = ix.writer()

        for idx, q in tqdm(self.questions_df.iterrows()):
            writer.add_document(
                question_id=q['question_id'],
                text=q['text'],
                answer=q['answer']
            )
        writer.commit()

    def embed(self, facts: List[Fact]) -> None:
        """
        Create embedding for facts and store in the `qrep` field of each fact.

        :param facts: the list of facts to be embedded.
        """
        texts = [c.text for c in facts]
        texts = (self.vocab.doc2bow(x) for x in nlp.pipe(texts))
        # need to set minimum_probability to a negative value
        # to prevent gensim output skipping topics
        doc_topic_dists = self.lda_model.get_document_topics(texts, minimum_probability=-1)
        for fact, dist in zip(facts, doc_topic_dists):
            # dist is something like [(d_i, i)]
            fact.qrep = np.asarray([d_i for i, d_i in dist])

    def get_fact(self, request: ScheduleRequest) -> Fact:
        """
        Get fact from database, insert if new.

        :param request: a `ScheduleRequest` where `fact_id` is used for database query.
        :return: a fact.
        """
        # retrieve from db if exists
        fact = self.db.get_fact(request.fact_id)
        if fact is not None:
            return fact

        fact = Fact(
            fact_id=request.fact_id,
            text=request.text,
            answer=request.answer,
            category=request.category,
            qrep=None,
            skill=None,
        )

        self.embed([fact])
        # the skill of a fact is an one-hot vector
        # the non-zero entry is a value between 0 and 1
        # indicating the average question accuracy
        fact.skill = np.zeros_like(fact.qrep)
        fact.skill[np.argmax(fact.qrep)] = 1
        fact.skill *= self.predict_one(fact)

        self.db.add_fact(fact)
        return fact

    def get_facts(self, requests: List[ScheduleRequest]) -> List[Fact]:
        """
        Get a list of facts from database based on a list of schedule requests.
        Insert new facst to database if any.
        Compared to calling `get_fact` for each request, this function embeds all new facts in one
        batch for faster speed.

        :param requests: the list of schedule requests whose `fact_id`s are used for query.
        :return: a list of facts.
        """
        new_facts, facts = [], []
        for i, r in enumerate(requests):
            fact = self.db.get_fact(r.fact_id)
            if fact is None:
                fact = Fact(
                    fact_id=r.fact_id,
                    text=r.text,
                    answer=r.answer,
                    category=r.category,
                    qrep=None,  # placeholder
                    skill=None  # placeholder
                )
                new_facts.append(fact)
            facts.append(fact)

        if len(new_facts) == 0:
            return facts

        logger.info('embed facts ' + str(len(new_facts)))
        self.embed(new_facts)

        fact_skills = self.get_skill_for_facts(new_facts)
        for i, fact in enumerate(new_facts):
            fact.skill = np.zeros_like(fact.qrep)
            fact.skill[np.argmax(fact.qrep)] = 1
            fact.skill *= fact_skills[i]
        self.db.add_facts(new_facts)
        return facts

    def get_user(self, user_id: str) -> User:
        """
        Get user from DB. If the user is new, we create a new user with default skill estimate and
        an empty study record.

        :param user_id: the `user_id` of the user to load.
        :return: the user.
        """
        # retrieve from db if exists
        user = self.db.get_user(user_id)
        if user is not None:
            return user

        # create new user and insert to db
        k = self.n_topics
        qrep = np.array([1 / k for _ in range(k)])
        new_user = User(
            user_id=user_id,
            category=None,
            qrep=[qrep],
            skill=[self.avg_user_skill_estimate]
        )
        self.db.add_user(new_user)
        return new_user

    # def retrieve(self, fact: dict) -> Tuple[List[dict], List[float]]:
    #     record_id = self.karl_to_question_id[int(fact['question_id'])]

    #     # 1. try to find in records with gameid-catnum-level
    #     if record_id in self.question_id_set:
    #         hits = self.questions_df[self.questions_df.question_id == record_id]
    #         if len(hits) > 0:
    #             facts = [fact.to_dict() for idx, fact in hits.iterrows()]
    #             return facts, [1 / len(hits) for _ in range(len(hits))]
    #     else:
    #         # return better default value without text search
    #         return 0.5

    #     # 2. do text search
    #     with self.ix.searcher() as searcher:
    #         query = QueryParser("text", self.ix.schema).parse(fact['text'])
    #         hits = searcher.search(query)
    #         hits = [x for x in hits if x['answer'] == fact['answer']]
    #         if len(hits) > 0:
    #             scores = [x.score for x in hits]
    #             ssum = np.sum(scores)
    #             scores = [x / ssum for x in scores]
    #             facts = [self.questions_df[self.questions_df['question_id'] == x['question_id']].iloc[0] for x in hits]
    #             return facts, scores

    #     # 3. not found
    #     return [], []

    def get_skill_for_fact(self, fact: Fact) -> float:
        # # 1. find same or similar fact in records
        # facts, scores = self.retrieve(fact)
        # if len(facts) > 0:
        #     return np.dot([x['prob'] for x in facts], scores)
        # # 2. use model to predict
        # return self.retention_model.predict_one(user, fact)
        # NOTE this is not in use
        return 0.5

    def get_skill_for_facts(self, facts: List[Fact]) -> List[float]:
        # NOTE this is not in use
        return [self.get_skill_for_fact(fact) for fact in facts]

    def set_params(self, params: Params):
        # TODO
        pass

    def dist_category(self, user: User, fact: Fact) -> float:
        """
        Penalize shift in predefined categories. 1 if fact category is different than the
        previous fact the user studied, 0 otherwise.
        
        :param user:
        :param fact:
        :return: 0 if same category, 1 if otherwise.
        """
        if user.category is None or fact.category is None:
            return 0
        return float(fact.category.lower() != user.category.lower())

    def dist_skill(self, user: User, fact: Fact) -> float:
        """
        Difference in the skill level between the user and the fact.
        Fact skill is a n_topics dimensional one-hot vector where the non-zero entry is the average
        question accuracy.
        User skill is the accumulated skill vectors of the max_queue previous facts.
        We also additionally penalize easier questions by a factor of 10.

        :param user:
        :param fact:
        :return: different in skill estimate, multiplied by 10 if fact is easier.
        """
        user_skill = self.get_discounted_average(user.skill, user.params.decay_qrep)
        user_skill = np.clip(user_skill, a_min=0.0, a_max=1.0)
        topic_idx = np.argmax(fact.qrep)
        d = fact.skill[topic_idx] - user_skill[topic_idx]
        # penalize easier questions by ten
        d *= 1 if d <= 0 else 10
        return abs(d)

    def dist_recall_batch(self, user: User, facts: List[Fact]) -> List[float]:
        """
        Penalize facts that we think the user cannot recall.
        Returns one minus the recall probability of each fact.
        This functions calls the retention model for all facts in one batch.

        :param user:
        :param facts:
        :return: the (1 - recall probablity) of each fact.
        """
        return (1 - self.retention_model.predict(user, facts)).tolist()

    def dist_cool_down(self, user: User, fact: Fact, date: datetime) -> float:
        """
        Avoid repetition of the same fact within a cool down time.
        We set a longer cool down time for correctly recalled facts.

        :param user:
        :param fact:
        :param date: current study date.
        :return: time in minutes till cool down time. 0 if passed.
        """
        prev_date, prev_response = user.previous_study.get(fact.fact_id, (None, None))
        if prev_date is None:
            return 0
        else:
            current_date = time.mktime(date.timetuple())
            prev_date = time.mktime(prev_date.timetuple())
            delta_minutes = max(float(current_date - prev_date) / 60, 0)
            # cool down is never negative
            if prev_response == 'correct':
                return max(user.params.cool_down_time_correct - delta_minutes, 0)
            else:
                return max(user.params.cool_down_time_wrong - delta_minutes, 0)

    def get_discounted_average(self, vectors: List[np.ndarray], decay: float) -> np.ndarray:
        """
        Compute the weighted average of a list of vectors with a geometric sequence of weights
        determined by `decay` (a value between 0 and 1).
        The last item has the initial (highest) weight.

        :param vectors: a list of numpy vectors to be averaged.
        :param decay: a real value between zero and one.
        :return: the weighted average vector.
        """
        if decay < 0 or decay > 1:
            raise ValueError('Decay rate for discounted average should be within (0, 1). \
                             Got {}'.format(decay))

        w = 1
        vs, ws = [], []
        for v in vectors[::-1]:
            vs.append(w * v)
            ws.append(w)
            w *= decay
        return np.sum(vs, axis=0) / np.sum(ws)

    def dist_qrep(self, user: User, fact: Fact) -> float:
        """
        Semantic similarity between user and fact.
        Cosine distance between the user's accumulated question representation and the fact's
        question representation.
        
        :param user:
        :param fact:
        :return: one minus cosine similarity.
        """
        def cosine_distance(a, b):
            return 1 - (a @ b.T) / (np.linalg.norm(a) * np.linalg.norm(b))
        user_qrep = self.get_discounted_average(user.qrep, user.params.decay_qrep)
        return cosine_distance(user_qrep, fact.qrep)

    def dist_leitner(self, user: User, fact: Fact, date: datetime) -> float:
        """
        Time till the scheduled date by Leitner measured by number of hours.
        The value can be negative when the fact is over-due in Leitner.

        :param user:
        :param fact:
        :return: distance in number of hours.
        """
        scheduled_date = user.leitner_scheduled_date.get(fact.fact_id, None)
        if scheduled_date is None:
            return 0
        else:
            # NOTE distance in hours, can be negative
            return (scheduled_date - date).total_seconds() / (60 * 60)

    def dist_sm2(self, user: User, fact: Fact, date: datetime) -> float:
        """
        Time till the scheduled date by SM-2 measured by number of hours.
        The value can be negative when the fact is over-due in SM-2.

        :param user:
        :param fact:
        :return: distance in number of hours.
        """
        scheduled_date = user.sm2_scheduled_date.get(fact.fact_id, None)
        if scheduled_date is None:
            return 0
        else:
            # NOTE distance in hours, can be negative
            return (scheduled_date - date).total_seconds() / (60 * 60)

    def score(self, user: User, facts: List[Fact], date: datetime) -> List[float]:
        """
        Compute the score between user and each fact, and compute the weighted sum distance.

        :param user:
        :param fact:
        :return: distance in number of hours.
        """
        recall_scores = self.dist_recall_batch(user, facts)
        scores = [{
            'qrep': self.dist_qrep(user, fact),
            'skill': self.dist_skill(user, fact),
            'recall': recall_scores[i],
            'category': self.dist_category(user, fact),
            'cool_down': self.dist_cool_down(user, fact, date),
            'leitner': self.dist_leitner(user, fact, date),
            'sm2': self.dist_sm2(user, fact, date),
        } for i, fact in enumerate(facts)]

        for i, _ in enumerate(scores):
            scores[i]['sum'] = sum([
                user.params.__dict__.get(key, 0) * value
                for key, value in scores[i].items()
            ])
        return scores

    def get_rationale(
            self,
            user: User,
            facts: List[Fact],
            date: datetime,
            scores: List[Dict[str, float]],
            order: List[int],
            top_n_facts=3
    ) -> str:
        """
        Create rationale HTML table for the top facts.

        :param user:
        :param facts:
        :param date: current study date passed to `schedule`.
        :param scores: the computed scores.
        :param order: the ordering of cards.
        :param top_n_facts: number of cards to explain.
        :return: an HTML table.
        """
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

        rr += '<h2>Top ranked facts</h2>'
        row_template = '<tr><td><b>{}</b></td> <td>{}</td></tr>'
        row_template_3 = '<tr><td><b>{}</b></td> <td>{:.4f} x {:.2f}</td></tr>'
        for i in order[:top_n_facts]:
            c = facts[i]
            prev_date, prev_response = user.previous_study.get(c.fact_id, ('-', '-'))
            rr += '<table style="float: left;">'
            rr += row_template.format('fact_id', c.fact_id)
            rr += row_template.format('answer', c.answer)
            rr += row_template.format('category', c.category)
            rr += row_template.format('topic', self.topic_words[np.argmax(c.qrep)])
            rr += row_template.format('prev_date', prev_date)
            rr += row_template.format('prev_response', prev_response)
            for k, v in scores[i].items():
                rr += row_template_3.format(k, v, user.params.__dict__.get(k, 0))
            rr += '<tr><td><b>{}</b></td> <td>{:.4f}</td></tr>'.format('sum', scores[i]['sum'])
            rr += row_template.format('ltn box', user.leitner_box.get(c.fact_id, '-'))
            rr += row_template.format('ltn dat', user.leitner_scheduled_date.get(c.fact_id, '-'))
            rr += row_template.format('sm2 rep', user.sm2_repetition.get(c.fact_id, '-'))
            rr += row_template.format('sm2 inv', user.sm2_interval.get(c.fact_id, '-'))
            rr += row_template.format('sm2 e_f', user.sm2_efactor.get(c.fact_id, '-'))
            rr += row_template.format('sm2 dat', user.sm2_scheduled_date.get(c.fact_id, '-'))
            rr += '</table>'
        return rr

    def get_facts_info(
            self,
            user: User,
            facts: List[Fact],
            date: datetime,
            scores: List[Dict[str, float]],
            order: List[int],
            top_n_facts=3
    ) -> List[Dict]:
        """
        Detailed ranking information for each card.

        :param user:
        :param facts:
        :param date: current study date passed to `schedule`.
        :param scores: the computed scores.
        :param order: the ordering of cards.
        :param top_n_facts: number of cards to explain.
        :return: an dictionary for each card.
        """
        facts_info = []
        for i in order[:top_n_facts]:
            fact_info = copy.deepcopy(facts[i].__dict__)
            fact_info['scores'] = scores[i]
            prev_date, prev_response = user.previous_study.get(fact_info['fact_id'], ('-', '-'))
            fact_info.update({
                'current date': date,
                'topic': self.topic_words[np.argmax(fact_info['qrep'])],
                'prev_response': prev_response,
                'prev_date': prev_date
            })
            fact_info.pop('qrep')
            fact_info.pop('skill')
            # fact_info.pop('results')
            facts_info.append(fact_info)
        return facts_info

    def rank_facts_for_user(
            self,
            user: User,
            facts: List[Fact],
            date: datetime,
            plot=False
    ) -> Dict[str, float]:
        """
        Score facts for user, rank them, and create rationale & card information.

        :param user:
        :param facts:
        :param date: current study date passed to `schedule`.
        :param plot: on-off switch for visualization.
        :return: everything.
        """
        scores = self.score(user, facts, date)
        order = np.argsort([s['sum'] for s in scores]).tolist()
        fact = facts[order[0]]

        user_qrep = self.get_discounted_average(user.qrep, user.params.decay_qrep)
        # plot_polar(user_qrep, '/fs/www-users/shifeng/temp/user.jpg', fill='green')
        # plot_polar(fact.qrep, '/fs/www-users/shifeng/temp/fact.jpg', fill='yellow')

        rationale = self.get_rationale(user, facts, date, scores, order)

        if plot:
            figname = '{}_{}_{}.jpg'.format(user.user_id, fact.fact_id, date.strftime('%Y-%m-%d-%H-%M'))
            local_filename = '/fs/www-users/shifeng/temp/' + figname
            remote_filename = 'http://users.umiacs.umd.edu/~shifeng/temp/' + figname
            self.plot_histogram(user_qrep, fact.qrep, local_filename)

            rationale += "</br>"
            rationale += "</br>"
            rationale += "<img src={}>".format(remote_filename)
            # rr += "<img src='http://users.umiacs.umd.edu/~shifeng/temp/fact.jpg'>"
            # rr += "<img src='http://users.umiacs.umd.edu/~shifeng/temp/user.jpg'>"

        facts_info = self.get_facts_info(user, facts, date, scores, order)
        output_dict = {
            'order': order,
            'rationale': rationale,
            'facts_info': facts_info,
            'scores': scores,
        }
        return output_dict

    def schedule(self, requests: List[ScheduleRequest], date: datetime, plot=False) -> dict:
        """
        The main schedule function.
        1. Load user and fact from database, insert if new.
            2.1. If precompute is off or no committed precompute schedule exists, compute schedule.
            2.2. Otherwise, load committed precompute schedule, score new cards if any and merge.
        4. If precompute is on, spin up two threads to compute the next-step schedule for both
           possible outcomes.
        5. Return the schedule.

        Note that we assume the list of requests only contains that of one user.
        
        :param requests: a list of scheduling requests containing both user and fact information.
        :param datetime: current study time.
        :param plot: on-off switch of visualizations.
        :return: a dictionary of everything about the schedule.
        """
        if len(requests) == 0:
            return [], [], ''

        schedule_profile = {}

        # mapping from user to scheduling requests
        # not used since we assume only one user
        user_to_requests = defaultdict(list)
        for i, request in enumerate(requests):
            user_to_requests[request.user_id].append(i)
        if len(user_to_requests) != 1:
            raise ValueError('Schedule only accpets 1 user. Received {}'.format(len(user_to_requests)))

        # load fact and user from db
        t0 = datetime.now()
        logger.info('scheduling {} facts'.format(len(requests)))
        facts = self.get_facts(requests)
        t1 = datetime.now()
        schedule_profile['get_facts'] = (len(requests), t1 - t0)

        user, indices = list(user_to_requests.items())[0]
        user = self.get_user(user)
        facts = [facts[i] for i in indices]

        t0 = datetime.now()
        if not self.precompute:
            return self.rank_facts_for_user(user, facts, date, plot=plot)
        t1 = datetime.now()
        schedule_profile['rank_facts_for_user (wo pre)'] = (len(facts), t1 - t0)

        # using precomputed schedules
        # read confirmed update & corresponding schedule
        if user.user_id not in self.precompute_commit:
            # no existing precomupted schedule, do regular schedule
            t0 = datetime.now()
            output_dict = self.rank_facts_for_user(user, facts, date, plot=plot)
            t1 = datetime.now()
            schedule_profile['rank_facts_for_user (pre not found)'] = (len(facts), t1 - t0)
        elif self.precompute_commit[user.user_id] == 'done':
            # precompute threads didn't finish before user responded and is marked as done by update
            # TODO this might be too conservative
            t0 = datetime.now()
            output_dict = self.rank_facts_for_user(user, facts, date, plot=plot)
            t1 = datetime.now()
            schedule_profile['rank_facts_for_user (pre marked done)'] = (len(facts), t1 - t0)
        else:
            # read precomputed schedule, check if new facts are added
            # if so, compute score for new facts, re-sort facts
            # no need to update previous fact / user
            # since updates does the commit after updating user & fact in db
            prev_results = self.precompute_commit[user.user_id]
            prev_fact_ids = [c.fact_id for c in prev_results['facts']]

            new_facts = []
            # prev_facts -> facts
            prev_fact_indices = {}
            # new_facts -> facts
            new_fact_indices = {}
            for i, c in enumerate(facts):
                if c.fact_id in prev_fact_ids:
                    prev_idx = prev_fact_ids.index(c.fact_id)
                    prev_fact_indices[prev_idx] = i
                else:
                    new_fact_indices[len(new_facts)] = i
                    new_facts.append(c)

            if len(new_fact_indices) + len(prev_fact_indices) != len(facts):
                raise ValueError('len(new_fact_indices) + len(prev_fact_indices) != len(facts)')

            t0 = datetime.now()
            # gather scores for both new and previous facts
            scores = [None] * len(facts)
            if len(new_facts) > 0:
                new_results = self.rank_facts_for_user(user, new_facts, date, plot=plot)
                for i, idx in new_fact_indices.items():
                    scores[idx] = new_results['scores'][i]
            for i, idx in prev_fact_indices.items():
                scores[idx] = prev_results['scores'][i]
            t1 = datetime.now()
            schedule_profile['rank_facts_for_user (new facts)'] = (len(new_facts), t1 - t0)

            t0 = datetime.now()
            order = np.argsort([s['sum'] for s in scores]).tolist()
            rationale = self.get_rationale(user, facts, date, scores, order)
            facts_info = self.get_facts_info(user, facts, date, scores, order)
            t1 = datetime.now()
            schedule_profile['get rationale and facts info'] = (len(facts), t1 - t0)

            output_dict = {
                'order': order,
                'scores': scores,
                'rationale': rationale,
                'facts_info': facts_info,
            }

        # output_dict generated
        fact_idx = output_dict['order'][0]

        if user.user_id in self.precompute_commit:
            # necessary to remove 'done' marked by update if exists
            self.precompute_commit.pop(user.user_id)

        # 'correct' branch
        thr_correct = threading.Thread(
            target=self.branch,
            args=(
                copy.deepcopy(user),
                copy.deepcopy(facts),
                date,
                fact_idx,
                'correct',
                plot,
            ),
            kwargs={}
        )
        thr_correct.start()
        # thr_correct.join()

        # 'wrong' branch
        thr_wrong = threading.Thread(
            target=self.branch,
            args=(
                copy.deepcopy(user),
                copy.deepcopy(facts),
                date,
                fact_idx,
                'wrong',
                plot,
            ),
            kwargs={}
        )
        thr_wrong.start()
        # thr_wrong.join()

        # self.branch(copy.deepcopy(user), copy.deepcopy(facts),
        #             date, fact_idx, 'correct', plot=plot)
        # self.branch(copy.deepcopy(user), copy.deepcopy(facts),
        #             date, fact_idx, 'wrong', plot=plot)

        output_dict['profile'] = schedule_profile
        return output_dict

    def branch(
            self,
            user: User,
            facts: List[Fact],
            date: datetime,
            fact_idx: int,
            response: str,
            plot=False
    ) -> None:
        """
        Compute a next-step schedule based on a guessed user response.
        1. Make copy of user and top fact.
        2. Mpdate (copied) user and top fact by (guessed) response but don't write to database.
        3. Compute next-step schedule with updated user and list of facts, where the top fact from
           previous schedule is replaced by the updated version.
        4. Store next-step schedule in the buffer, wait for an `update` call with the actual user
           response to commit.

        Note that the dates used in both update and score are not accurate since we don't know when
        the user will respond and request for the next fact. But it should affect all facts (with
        non-zero repetition) equally.

        :param user:
        :param facts:
        :param date: current study time. Note that this usually is not the actual date of the next
                     schedule request.
        :param fact_idx: index of the top fact from previous schedule, which is current being shown
                         to the user.
        :param response: guessed response.
        :param plot: on-off switch for visualizations.
        """
        self.update_user_fact(user, facts[fact_idx], date, response)
        results = self.rank_facts_for_user(user, facts, date, plot=plot)
        results.update({
            'user': user,
            'fact': facts[fact_idx],
            'facts': facts,
        })
        if self.precompute_commit.get(user.user_id, None) != 'done':
            # if done, user already responded, marked as done by update
            self.precompute_future[response][user.user_id] = results

    def update_user_fact(self, user: User, fact: Fact, date: datetime, response: str) -> None:
        """
        Update the user with a response on this fact.
        
        :param user:
        :param fact:
        :param date: date on which user studied fact.
        :param response: user's response on this fact.
        """
        # update qrep
        user.qrep.append(fact.qrep)
        if len(user.qrep) >= user.params.max_queue:
            user.qrep.pop(0)

        # update skill
        if response:
            user.skill.append(fact.skill)
            if len(user.skill) >= user.params.max_queue:
                # queue is full, remove oldest entry
                user.skill.pop(0)

        user.category = fact.category
        user.previous_study[fact.fact_id] = (date, response)

        # update retention features
        fact.results.append(response == 'correct')
        user.results.append(response == 'correct')
        if fact.fact_id not in user.count_correct_before:
            user.count_correct_before[fact.fact_id] = 0
        if fact.fact_id not in user.count_wrong_before:
            user.count_wrong_before[fact.fact_id] = 0
        if response:
            user.count_correct_before[fact.fact_id] += 1
        else:
            user.count_wrong_before[fact.fact_id] += 1

        self.leitner_update(user, fact, response)
        self.sm2_update(user, fact, response)

    def update(self, requests: List[ScheduleRequest], date: datetime) -> dict:
        """
        The main update function.
        1. Read user and facts from database.
        2. Create history entry.
            3.1. If precompute is off, update user and fact
            3.2. If otherwise but precompute scheduling is not done, mark it as obsolete.
            3.3. If scheduling is finished, commit the one corresponding to the actual response.
        4. Update user and fact in database.
        5. Return update details.

        :param requests:
        :param date: date of the study.
        :return: details.
        """
        # mapping from user to fact indices
        # not used since we assume only one user
        user_to_requests = defaultdict(list)
        for i, request in enumerate(requests):
            user_to_requests[request.user_id].append(i)
        facts = self.get_facts(requests)

        if len(user_to_requests) != 1:
            raise ValueError('Update only accpets 1 user. Received {}'.format(len(user_to_requests)))
        user, indices = list(user_to_requests.items())[0]
        user = self.get_user(user)

        if len(indices) != 1:
            raise ValueError('Update only accpets 1 fact. Received {}'.format(len(indices)))
        request = requests[indices[0]]
        fact = facts[indices[0]]

        # find that temporary history entry and update
        # temp_history_id = json.dumps({'user_id': user.user_id, 'fact_id': fact.fact_id})
        # history = self.db.get_history(temp_history_id)
        # if history is not None:
        if False:
            pass
        #     history.__dict__.update({
        #         'history_id': request.history_id,
        #         'response': request.label,
        #         'judgement': request.label,
        #         'date': date
        #     })
        #     self.db.update_history(temp_history_id, history)
        else:
            history = History(
                history_id=request.history_id,
                user_id=request.user_id,
                fact_id=fact.fact_id,
                response=request.label,
                judgement=request.label,
                user_snapshot=json.dumps(user.pack()),
                scheduler_snapshot=json.dumps(user.params.__dict__),
                fact_ids=json.dumps([x.fact_id for x in facts]),
                scheduler_output='',
                date=date)
            self.db.add_history(history)

        # for detail display, this need to happen before the `update_user_fact` below
        old_user = copy.deepcopy(user)

        # commit 
        if not self.precompute:
            self.update_user_fact(user, fact, date, request.label)
        elif user.user_id not in self.precompute_future[request.label]:
            self.update_user_fact(user, fact, date, request.label)
            # NOTE precompute did not finish before user responded
            # mark commit as taken
            self.precompute_commit[user.user_id] = 'done'
        else:
            results = self.precompute_future[request.label][user.user_id]
            self.precompute_commit[user.user_id] = results
            user = results['user']
            fact = results['fact']

        self.db.update_user(user)
        self.db.update_fact(fact)

        detail = {
            'response': request.label,
            'old ltn box': old_user.leitner_box.get(fact.fact_id, '-'),
            'new ltn box': user.leitner_box.get(fact.fact_id, '-'),
            'old ltn dat': str(old_user.leitner_scheduled_date.get(fact.fact_id, '-')),
            'new ltn dat': str(user.leitner_scheduled_date.get(fact.fact_id, '-')),
            'old sm2 rep': old_user.sm2_repetition.get(fact.fact_id, '-'),
            'new sm2 rep': user.sm2_repetition.get(fact.fact_id, '-'),
            'old sm2 inv': old_user.sm2_interval.get(fact.fact_id, '-'),
            'new sm2 inv': user.sm2_interval.get(fact.fact_id, '-'),
            'old sm2 e_f': old_user.sm2_efactor.get(fact.fact_id, '-'),
            'new sm2 e_f': user.sm2_efactor.get(fact.fact_id, '-'),
            'old sm2 dat': str(old_user.sm2_scheduled_date.get(fact.fact_id, '-')),
            'new sm2 dat': str(user.sm2_scheduled_date.get(fact.fact_id, '-')),
        }

        return detail

    def leitner_update(self, user: User, fact: Fact, response: str) -> None:
        """
        Update Leitner box and scheduled date of card.

        :param user:
        :param fact:
        :param response: 'correct' or 'wrong'.
        """
        # leitner boxes 1~10
        # days[0] = None as placeholder since we don't have box 0
        # days[9] and days[10] = 9999 to make it never repeat
        days = [None, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 9999, 9999]
        increment_days = {i: x for i, x in enumerate(days)}

        # boxes: 1 ~ 10
        cur_box = user.leitner_box.get(fact.fact_id, None)
        if cur_box is None:
            cur_box = 1
        new_box = cur_box + (1 if response == 'correct' else -1)
        new_box = max(min(new_box, 10), 1)
        user.leitner_box[fact.fact_id] = new_box
        interval = timedelta(days=increment_days[new_box])
        # TODO is this correct? increment on previous instead of current study date?
        prev_date, prev_response = user.previous_study[fact.fact_id]
        user.leitner_scheduled_date[fact.fact_id] = prev_date + interval

    def sm2_update(self, user: User, fact: Fact, response: str) -> None:
        """
        Update SM-2 e_factor, repetition, and interval.
        
        :param user:
        :param fact:
        :param response:
        """
        def get_quality_from_response(response: str) -> int:
            return 4 if response == 'correct' else 1

        e_f = user.sm2_efactor.get(fact.fact_id, 2.5)
        inv = user.sm2_interval.get(fact.fact_id, 1)
        rep = user.sm2_repetition.get(fact.fact_id, 0) + 1

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

        user.sm2_repetition[fact.fact_id] = rep
        user.sm2_efactor[fact.fact_id] = e_f
        user.sm2_interval[fact.fact_id] = inv
        prev_date, prev_response = user.previous_study[fact.fact_id]
        user.sm2_scheduled_date[fact.fact_id] = prev_date + timedelta(days=inv)

    def plot_histogram(self, user_qrep: np.ndarray, fact_qrep: np.ndarray, filename: str) -> None:
        """
        Visualize the topic distribution, overlap user with fact.

        :param fact_qrep: question representation of the fact.
        :param user_qrep: the accumulated question representation of the user.
        :param filename: save figure to this path.
        """
        max_qrep = np.max((fact_qrep, user_qrep), axis=0)
        top_topics = np.argsort(-max_qrep)[:10]
        fact_qrep = np.array(fact_qrep)[top_topics].tolist()
        user_qrep = np.array(user_qrep)[top_topics].tolist()
        top_topic_words = [self.topic_words[i] for i in top_topics]
        topic_type = CategoricalDtype(categories=top_topic_words, ordered=True)
        df = pd.DataFrame({
            'topics': top_topic_words * 2,
            'weight': fact_qrep + user_qrep,
            'label': ['fact' for _ in top_topics] + ['user' for _ in top_topics]
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
        p.save(filename, verbose=False)
