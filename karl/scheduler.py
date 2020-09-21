#!/usr/bin/env python
# coding: utf-8

import os
import json
import time
import copy
import pickle
import gensim
import hashlib
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

from karl.lda import process_question
from karl.new_util import ScheduleRequest, Params, User, Fact, Record, UserStat
from karl.new_util import parse_date, theme_fs
from karl.retention.baseline import RetentionModel
# from karl.new_retention import HFRetentionModel as RetentionModel


CORRECT = True
WRONG = False


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
            preemptive=True,
            lda_dir='checkpoints/gensim_quizbowl_10_1585102364.5221019',
            whoosh_index='whoosh_index',
    ) -> None:
        """
        :param db_filename: location of database store.
        :param preemptive: on-off switch of preemptive scheduling.
        :param lda_dir: gensim LDA model directory.
        :param whoosh_index: whoosh index for text matching.
        """
        self.preemptive = preemptive
        self.lda_dir = lda_dir
        self.whoosh_index = whoosh_index

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
        # df = df[df[CORRECT].notna()]
        # df[CORRECT] = df[CORRECT].apply(lambda x: 1 if x == 1 else 0)
        # df_grouped = df.reset_index()[['question_id', CORRECT]].groupby('question_id')
        # dict_correct_mean = df_grouped.mean().to_dict()[CORRECT]
        # dict_records_cnt = df_grouped.count().to_dict()[CORRECT]
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
        self.preemptive_future = {CORRECT: {}, WRONG: {}}
        self.preemptive_commit = dict()

        # user_id -> random hash string that corresponds to the card being shown to the user
        self.debug_id = dict()

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
        facts = [
            Fact(
                fact_id=c['fact_id'],
                text=c['text'],
                answer=c['answer'],
                category=c['category'],
                qrep=None,
                skill=None)
            for c in facts
        ]

        self.embed(facts)
        qreps = [c.qrep for c in facts]
        estimates = [[] for _ in range(self.n_topics)]
        for fact, qrep in zip(facts, qreps):
            topic_idx = np.argmax(qrep)
            estimates[topic_idx].append(self.get_skill_for_fact(fact))
        estimates = [np.mean(x) for x in estimates]
        with open(estimate_file_dir, 'w') as f:
            for e in estimates:
                f.write(str(e) + '\n')
        return np.array(estimates)

    def reset_user(self, session, user_id=None) -> None:
        """
        Delete a specific user (when `user_id` is provided) or all users (when `user_id` is `None`)
        and the corresponding history entries from database. This resets the study progress.

        :param user_id: the `user_id` of the user to be reset. Resets all users if `None`.
        """

        if user_id is not None:
            # remove user from db
            user = session.query(User).get(user_id)
            if user is not None:
                session.delete(user)

            if user_id in self.preemptive_future[CORRECT]:
                self.preemptive_future[CORRECT].pop(user_id)
            if user_id in self.preemptive_future[WRONG]:
                self.preemptive_future[WRONG].pop(user_id)
            if user_id in self.preemptive_commit:
                self.preemptive_commit.pop(user_id)
        else:
            # remove all users
            session.query(User).delete()

    def reset_fact(self, session, fact_id=None) -> None:
        """
        Delete a specific fact (if `fact_id` is provided) or all facts (if `fact_id` is None) from
        database. This removed the cached embedding and skill estimate of the fact(s).

        :param fact_id: the `fact_id` of the fact to be reset. Resets all facts if `None`.
        """
        fact = session.query(Fact).get(fact_id)
        if fact is not None:
            session.query(Fact).delete(fact)

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

    def get_fact(self, session, request: ScheduleRequest) -> Fact:
        """
        Get fact from database, insert if new.

        :param request: a `ScheduleRequest` where `fact_id` is used for database query.
        :return: a fact.
        """
        # retrieve from db if exists
        fact = session.query(Fact).get(request.fact_id)
        if fact is not None:
            return fact

        fact = Fact(
            fact_id=request.fact_id,
            text=request.text,
            answer=request.answer,
            category=request.category,
            deck_name=request.deck_name,
            deck_id=request.deck_id,
            qrep=None,
            skill=None,
        )

        self.embed([fact])
        # the skill of a fact is an one-hot vector
        # the non-zero entry is a value between 0 and 1
        # indicating the average question accuracy
        fact.skill = np.zeros_like(fact.qrep)
        fact.skill[np.argmax(fact.qrep)] = 1
        fact.skill *= self.get_skill_for_fact(fact)

        session.add(fact)
        session.commit()
        return fact

    def get_facts(self, session, requests: List[ScheduleRequest]) -> List[Fact]:
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
            fact = session.query(Fact).get(r.fact_id)
            if fact is None:
                fact = Fact(
                    fact_id=r.fact_id,
                    text=r.text,
                    answer=r.answer,
                    category=r.category,
                    deck_name=r.deck_name,
                    deck_id=r.deck_id,
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
        session.bulk_save_objects(new_facts)
        session.commit()
        return facts

    def get_all_users(self, session) -> List[User]:
        return session.query(User).all()

    def get_user(self, session, user_id: str) -> User:
        """
        Get user from DB. If the user is new, we create a new user with default skill estimate and
        an empty study record.

        :param user_id: the `user_id` of the user to load.
        :return: the user.
        """
        # retrieve from db if exists
        user = session.query(User).get(user_id)
        if user is not None:
            return user

        # create new user and insert to db
        new_user = User(user_id=user_id)
        session.add(new_user)
        session.commit()
        return new_user

    def get_records(self, session, user_id: str, deck_id: str = None,
                    date_start: str = None, date_end: str = None):
        if date_start is None:
            date_start = '2008-06-11 08:00:00'
        if date_end is None:
            date_end = '2038-06-11 08:00:00'

        date_start = parse_date(date_start)
        date_end = parse_date(date_end)
        records = session.query(Record).filter(Record.user_id == user_id).\
            filter(Record.date >= date_start, Record.date <= date_end)
        if deck_id is not None:
            records = records.filter(Record.deck_id == deck_id)
        return records.all()

    def get_user_stats(self, session, user_id: str, deck_id: str = None,
                       date_start: str = None, date_end: str = None):
        if date_start is None:
            date_start = '2008-06-11 08:00:00'
        if date_end is None:
            date_end = '2038-06-11 08:00:00'

        date_start = parse_date(date_start).date()
        date_end = parse_date(date_end).date() + timedelta(days=1)  # TODO temporary fix, wait for Matthew

        if deck_id is None:
            deck_id = 'all'

        # last record no later than start date
        before_stat = session.query(UserStat).\
            filter(UserStat.user_id == user_id).\
            filter(UserStat.deck_id == deck_id).\
            filter(UserStat.date < date_start).order_by(UserStat.date.desc()).first()
        # last record no later than end date
        after_stat = session.query(UserStat).\
            filter(UserStat.user_id == user_id).\
            filter(UserStat.deck_id == deck_id).\
            filter(UserStat.date <= date_end).order_by(UserStat.date.desc()).first()

        if after_stat is None or after_stat.date < date_start:
            return {
                'new_facts': 0,
                'reviewed_facts': 0,
                'new_correct': 0,
                'reviewed_correct': 0,
                'total_seen': 0,
                'total_milliseconds': 0,
                'total_seconds': 0,
                'total_minutes': 0,
                'elapsed_milliseconds_text': 0,
                'elapsed_milliseconds_answer': 0,
                'elapsed_seconds_text': 0,
                'elapsed_seconds_answer': 0,
                'elapsed_minutes_text': 0,
                'elapsed_minutes_answer': 0,
                'known_rate': 0,
                'new_known_rate': 0,
                'review_known_rate': 0,
            }

        if before_stat is None:
            user_stat_id = json.dumps({
                'user_id': user_id,
                'date': str(date_start),
                'deck_id': deck_id,
            })
            before_stat = UserStat(
                user_stat_id=user_stat_id,
                user_id=user_id,
                deck_id=deck_id,
                date=date_start,
                new_facts=0,
                reviewed_facts=0,
                new_correct=0,
                reviewed_correct=0,
                total_seen=0,
                total_milliseconds=0,
                total_seconds=0,
                total_minutes=0,
                elapsed_milliseconds_text=0,
                elapsed_milliseconds_answer=0,
                elapsed_seconds_text=0,
                elapsed_seconds_answer=0,
                elapsed_minutes_text=0,
                elapsed_minutes_answer=0,
            )

        total_correct = (after_stat.new_correct + after_stat.reviewed_correct) - (before_stat.new_correct + before_stat.reviewed_correct)

        known_rate = 0
        if after_stat.total_seen > before_stat.total_seen:
            known_rate = total_correct / (after_stat.total_seen - before_stat.total_seen)

        new_known_rate = 0
        if after_stat.new_facts > before_stat.new_facts:
            new_known_rate = (after_stat.new_correct - before_stat.new_correct) / (after_stat.new_facts - before_stat.new_facts)

        review_known_rate = 0
        if after_stat.reviewed_facts > before_stat.reviewed_facts:
            review_known_rate = (after_stat.reviewed_correct - before_stat.reviewed_correct) / (after_stat.reviewed_facts - before_stat.reviewed_facts)

        return {
            'new_facts': after_stat.new_facts - before_stat.new_facts,
            'reviewed_facts': after_stat.reviewed_facts - before_stat.reviewed_facts,
            'new_correct': after_stat.new_correct - before_stat.new_correct,
            'reviewed_correct': after_stat.reviewed_correct - before_stat.reviewed_correct,
            'total_seen': after_stat.total_seen - before_stat.total_seen,
            'total_milliseconds': after_stat.total_milliseconds - before_stat.total_milliseconds,
            'total_seconds': after_stat.total_seconds - before_stat.total_seconds,
            'total_minutes': after_stat.total_minutes - before_stat.total_minutes,
            'elapsed_milliseconds_text': after_stat.elapsed_milliseconds_text - before_stat.elapsed_milliseconds_text,
            'elapsed_milliseconds_answer': after_stat.elapsed_milliseconds_answer - before_stat.elapsed_milliseconds_answer,
            'elapsed_seconds_text': after_stat.elapsed_seconds_text - before_stat.elapsed_seconds_text,
            'elapsed_seconds_answer': after_stat.elapsed_seconds_answer - before_stat.elapsed_seconds_answer,
            'elapsed_minutes_text': after_stat.elapsed_minutes_text - before_stat.elapsed_minutes_text,
            'elapsed_minutes_answer': after_stat.elapsed_minutes_answer - before_stat.elapsed_minutes_answer,
            'known_rate': round(known_rate * 100, 2),
            'new_known_rate': round(new_known_rate * 100, 2),
            'review_known_rate': round(review_known_rate * 100, 2),
        }

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
        # NOTE this is not in use
        # # 1. find same or similar fact in records
        # facts, scores = self.retrieve(fact)
        # if len(facts) > 0:
        #     return np.dot([x['prob'] for x in facts], scores)
        # # 2. use model to predict
        # return self.retention_model.predict_one(user, fact)
        return 0.5

    def get_skill_for_facts(self, facts: List[Fact]) -> List[float]:
        # NOTE this is not in use
        return [self.get_skill_for_fact(fact) for fact in facts]

    def set_user_params(self, session, user_id: str, params: Params):
        """
        Set parameters for user.

        :param params:
        """
        user = self.get_user(session, user_id)
        user.params = params
        session.commit()

    def dist_category(self, user: User, fact: Fact) -> float:
        """
        Penalize shift in predefined categories. 1 if fact category is different than the
        previous fact the user studied, 0 otherwise.

        :param user:
        :param fact:
        :return: 0 if same category, 1 if otherwise.
        """
        if len(user.records) == 0:
            return 0
        last_fact = user.records[-1].fact
        if last_fact is None or last_fact.category is None or fact.category is None:
            return 0
        return float(fact.category != last_fact.category)

    def dist_answer(self, user: User, fact: Fact) -> float:
        """
        Penalize repetition of the same answer.
        If the same answer appeared T cards ago, penalize by 1 / T

        :param user:
        :param fact:
        :return: 1 if same answer, 0 if otherwise.
        """
        if (
                len(user.records) == 0
                or fact.answer is None
        ):
            return 0
        T = 0
        for i, record in enumerate(user.records[::-1]):
            # from most recent to least recent
            if record.fact.answer == fact.answer:
                T = i + 1
                break
        if T == 0:
            return 0
        else:
            return 1 / T

    def dist_skill(self, user: User, fact: Fact) -> float:
        """
        Difference in the skill level between the user and the fact.
        Fact skill is a n_topics dimensional one-hot vector where the non-zero entry is the average
        question accuracy.
        User skill is the accumulated skill vectors of the max_recent_facts previous facts.
        We also additionally penalize easier questions by a factor of 10.

        :param user:
        :param fact:
        :return: different in skill estimate, multiplied by 10 if fact is easier.
        """
        if len(user.records) == 0:
            user_skill = copy.deepcopy(self.avg_user_skill_estimate)
        else:
            recent_facts = [record.fact for record in user.records[::-1][:user.params.max_recent_facts]]
            user_skill = self.get_discounted_average([fact.skill for fact in recent_facts],
                                                     user.params.decay_skill)
            user_skill = np.clip(user_skill, a_min=0.0, a_max=1.0)
        topic_idx = np.argmax(fact.qrep)
        d = fact.skill[topic_idx] - user_skill[topic_idx]
        # penalize easier questions by ten
        d *= 1 if d <= 0 else 10
        return abs(d)

    def dist_recall_batch(self, user: User, facts: List[Fact], date: datetime) -> List[float]:
        """
        With recall_target = 1, we basically penalize facts that the user
        likely cannot cannot recall.
        With recall_target < 1, we look for facts whose recall probability is
        close to the designated target.

        Returns one minus the recall probability of each fact.
        This functions calls the retention model for all facts in one batch.

        :param user:
        :param facts:
        :return: the (recall_target - recall probablity) of each fact.
        """
        target = user.params.recall_target
        p_of_recall = self.retention_model.predict(user, facts, date)
        return np.abs(target - p_of_recall).tolist()

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
            # TODO handle correct on first try
            return 0
        else:
            if isinstance(prev_date, str):
                prev_date = parse_date(prev_date)
            current_date = time.mktime(date.timetuple())
            prev_date = time.mktime(prev_date.timetuple())
            delta_minutes = max(float(current_date - prev_date) / 60, 0)
            # cool down is never negative
            if prev_response == CORRECT:
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
        if len(user.records) == 0:
            user_qrep = np.array([1 / self.n_topics for _ in range(self.n_topics)])
        else:
            recent_facts = [record.fact for record in user.records[::-1][:user.params.max_recent_facts]]
            user_qrep = self.get_discounted_average([x.qrep for x in recent_facts],
                                                    user.params.decay_qrep)
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
            if isinstance(scheduled_date, str):
                scheduled_date = parse_date(scheduled_date)
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
            if isinstance(scheduled_date, str):
                scheduled_date = parse_date(scheduled_date)
            # NOTE distance in hours, can be negative
            return (scheduled_date - date).total_seconds() / (60 * 60)

    def score(self, user: User, facts: List[Fact], date: datetime) -> List[Dict[str, float]]:
        """
        Compute the score between user and each fact, and compute the weighted sum distance.

        :param user:
        :param fact:
        :return: distance in number of hours.
        """
        recall_scores = self.dist_recall_batch(user, facts, date)
        scores = [{
            'qrep': self.dist_qrep(user, fact),
            # 'skill': self.dist_skill(user, fact),
            'recall': recall_scores[i],
            'category': self.dist_category(user, fact),
            'answer': self.dist_answer(user, fact),
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
            top_n_facts: int = 3
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

        dump_dict = {
            'user_id': user.user_id,
            'fact_id': facts[order[0]].fact_id,
            'date': date.strftime('%Y-%m-%dT%H:%M:%S%z'),
        }
        debug_id = hashlib.md5(json.dumps(dump_dict).encode('utf8')).hexdigest()
        self.debug_id[user.user_id] = debug_id

        rr += '<h2>Debug ID: {}</h2>'.format(debug_id)
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
    ) -> dict:
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

        # if len(user.records) == 0:
        #     user_qrep = np.array([1 / self.n_topics for _ in range(self.n_topics)])
        # else:
        #     recent_facts = [record.fact for record in user.records[::-1][:user.params.max_recent_facts]]
        #     user_qrep = self.get_discounted_average([x.qrep for x in recent_facts],
        #                                             user.params.decay_qrep)
        # plot_polar(user_qrep, '/fs/www-users/shifeng/temp/user.jpg', fill='green')
        # plot_polar(fact.qrep, '/fs/www-users/shifeng/temp/fact.jpg', fill='yellow')

        rationale = self.get_rationale(user, facts, date, scores, order)

        if plot:
            figname = '{}_{}_{}.jpg'.format(user.user_id, fact.fact_id, date.strftime('%Y-%m-%d-%H-%M'))
            # local_filename = '/fs/www-users/shifeng/temp/' + figname
            remote_filename = 'http://users.umiacs.umd.edu/~shifeng/temp/' + figname
            # self.plot_histogram(user_qrep, fact.qrep, local_filename)

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

    def schedule(self, session, requests: List[ScheduleRequest], date: datetime, plot=False) -> dict:
        """
        The main schedule function.
        1. Load user and fact from database, insert if new.
            2.1. If preemptive is off or no committed preemptive schedule exists, compute schedule.
            2.2. Otherwise, load committed preemptive schedule, score new cards if any and merge.
        4. If preemptive is on, spin up two threads to compute the next-step schedule for both
           possible outcomes.
        5. Return the schedule.

        Note that we assume the list of requests only contains that of one user.

        :param requests: a list of scheduling requests containing both user and fact information.
        :param datetime: current study time.
        :param plot: on-off switch of visualizations.
        :return: a dictionary of everything about the schedule.
        """
        if len(requests) == 0:
            return {}

        # time the scheduler
        schedule_timing_profile = {}

        # mapping from user to scheduling requests
        # not used since we assume only one user
        user_to_requests = defaultdict(list)
        for i, request in enumerate(requests):
            user_to_requests[request.user_id].append(i)
        if len(user_to_requests) != 1:
            raise ValueError('Schedule only accpets 1 user. Received {}'.format(len(user_to_requests)))

        # load fact and user from db
        t0 = datetime.now()
        facts = self.get_facts(session, requests)
        t1 = datetime.now()
        schedule_timing_profile['get_facts'] = (len(requests), t1 - t0)

        user, indices = list(user_to_requests.items())[0]
        user = self.get_user(session, user)

        if requests[0].repetition_model is None:
            repetition_model_name = 'default'
        else:
            repetition_model_name = requests[0].repetition_model.lower()

        if repetition_model_name == 'sm2' or repetition_model_name == 'sm-2':
            user.params = Params(
                qrep=0,
                skill=0,
                recall=0,
                category=0,
                leitner=0,
                sm2=1,
            )
        elif repetition_model_name == 'leitner':
            user.params = Params(
                qrep=0,
                skill=0,
                recall=0,
                category=0,
                leitner=1,
                sm2=0,
            )
        elif repetition_model_name == 'karl':
            if user.params.qrep == 0:
                # user might have specified some of the params
                # only reset to karl parameters if currently using non-karl scheduler
                # in which case params.qrep should be 0
                user.params = Params()
        elif repetition_model_name == 'karl50':
            if user.params.qrep == 0:
                # user might have specified some of the params
                # only reset to karl parameters if currently using non-karl scheduler
                # in which case params.qrep should be 0
                user.params = Params(recall_target=0.5)
        elif repetition_model_name == 'karl85':
            if user.params.qrep == 0:
                # user might have specified some of the params
                # only reset to karl parameters if currently using non-karl scheduler
                # in which case params.qrep should be 0
                user.params = Params(recall_target=0.85)

        facts = [facts[i] for i in indices]

        if not self.preemptive:
            return self.rank_facts_for_user(user, facts, date, plot=plot)
        t1 = datetime.now()

        # using preemptived schedules
        # read confirmed update & corresponding schedule
        if user.user_id not in self.preemptive_commit:
            # no existing precomupted schedule, do regular schedule
            t0 = datetime.now()
            output_dict = self.rank_facts_for_user(user, facts, date, plot=plot)
            t1 = datetime.now()
            schedule_timing_profile['rank_facts_for_user (pre not found)'] = (len(facts), t1 - t0)
        elif self.preemptive_commit[user.user_id] == 'done':
            # preemptive threads didn't finish before user responded and is marked as done by update
            # TODO this might be too conservative
            t0 = datetime.now()
            output_dict = self.rank_facts_for_user(user, facts, date, plot=plot)
            t1 = datetime.now()
            schedule_timing_profile['rank_facts_for_user (pre marked done)'] = (len(facts), t1 - t0)
        else:
            # read preemptived schedule, check if new facts are added
            # if so, compute score for new facts, re-sort facts
            # no need to update previous fact / user
            # since updates does the commit after updating user & fact in db
            prev_results = self.preemptive_commit[user.user_id]
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
            schedule_timing_profile['rank_facts_for_user (new facts)'] = (len(new_facts), t1 - t0)

            t0 = datetime.now()
            order = np.argsort([s['sum'] for s in scores]).tolist()
            rationale = self.get_rationale(user, facts, date, scores, order)
            facts_info = self.get_facts_info(user, facts, date, scores, order)
            t1 = datetime.now()
            schedule_timing_profile['get rationale and facts info'] = (len(facts), t1 - t0)

            output_dict = {
                'order': order,
                'scores': scores,
                'rationale': rationale,
                'facts_info': facts_info,
            }

        # output_dict generated
        fact_idx = output_dict['order'][0]

        if user.user_id in self.preemptive_commit:
            # necessary to remove 'done' marked by update if exists
            self.preemptive_commit.pop(user.user_id)

        # CORRECT branch
        thr_correct = threading.Thread(
            target=self.branch,
            args=(
                copy.deepcopy(user),
                copy.deepcopy(facts),
                date,
                fact_idx,
                CORRECT,
                plot,
            ),
            kwargs={}
        )
        thr_correct.start()
        # thr_correct.join()

        # WRONG branch
        thr_wrong = threading.Thread(
            target=self.branch,
            args=(
                copy.deepcopy(user),
                copy.deepcopy(facts),
                date,
                fact_idx,
                WRONG,
                plot,
            ),
            kwargs={}
        )
        thr_wrong.start()
        # thr_wrong.join()

        # self.branch(copy.deepcopy(user), copy.deepcopy(facts),
        #             date, fact_idx, CORRECT, plot=plot)
        # self.branch(copy.deepcopy(user), copy.deepcopy(facts),
        #             date, fact_idx, WRONG, plot=plot)

        output_dict['profile'] = schedule_timing_profile
        logger.info('scheduled fact {}'.format(facts[fact_idx].answer))
        profile_str = [
            '{}: {} ({})'.format(k, v, c)
            for k, (c, v) in schedule_timing_profile.items()
        ]
        logger.info('\n' + '\n'.join(profile_str))

        return output_dict

    def branch(
            self,
            user: User,
            facts: List[Fact],
            date: datetime,
            fact_idx: int,
            response: bool,
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
        pre_schedule = {
            'order': results['order'],
            'rationale': results['rationale'],
            'facts_info': results['facts_info'],
            'scores': results['scores'],
            'user': user,
            'fact': facts[fact_idx],
            'facts': facts,
        }
        if self.preemptive_commit.get(user.user_id, None) != 'done':
            # if done, user already responded, marked as done by update
            self.preemptive_future[response][user.user_id] = pre_schedule

    def update_user_fact(self, user: User, fact: Fact, date: datetime, response: bool) -> None:
        """
        Update the user with a response on this fact.

        :param user:
        :param fact:
        :param date: date on which user studied fact.
        :param response: user's response on this fact.
        """
        # update category and previous study (date and response)
        user.previous_study[fact.fact_id] = (str(date), response)

        # update retention features
        fact.results.append(response == CORRECT)
        user.results.append(response == CORRECT)
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

    def update(self, session, requests: List[ScheduleRequest], date: datetime) -> dict:
        """
        The main update function.
        1. Read user and facts from database.
        2. Create history entry.
            3.1. If preemptive is off, update user and fact
            3.2. If otherwise but preemptive scheduling is not done, mark it as obsolete.
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
        facts = self.get_facts(session, requests)

        if len(user_to_requests) != 1:
            raise ValueError('Update only accpets 1 user. Received {}'.format(len(user_to_requests)))
        user, indices = list(user_to_requests.items())[0]
        user = self.get_user(session, user)

        if len(indices) != 1:
            raise ValueError('Update only accpets 1 fact. Received {}'.format(len(indices)))
        request = requests[indices[0]]
        fact = facts[indices[0]]

        user_snapshot = {
            'leitner_box': user.leitner_box,
            'count_correct_before': user.count_correct_before,
            'count_wrong_before': user.count_wrong_before,
        }

        record = Record(
            record_id=request.history_id,
            debug_id=self.debug_id.get(request.user_id, 'null'),
            user_id=request.user_id,
            fact_id=fact.fact_id,
            deck_id=fact.deck_id,
            response=request.label,
            judgement=request.label,
            user_snapshot=json.dumps(user_snapshot),
            scheduler_snapshot=json.dumps(user.params.__dict__),
            fact_ids=json.dumps([x.fact_id for x in facts]),
            scheduler_output='',
            elapsed_seconds_text=request.elapsed_seconds_text,
            elapsed_seconds_answer=request.elapsed_seconds_answer,
            elapsed_milliseconds_text=request.elapsed_milliseconds_text,
            elapsed_milliseconds_answer=request.elapsed_milliseconds_answer,
            is_new_fact=int(fact.fact_id not in user.previous_study),
            date=date,
        )
        session.add(record)
        session.commit()

        # for detail display, this need to happen before the `update_user_fact` below
        # old_user = copy.deepcopy(user)

        # commit
        if not self.preemptive:
            self.update_user_fact(user, fact, date, request.label)
        elif user.user_id not in self.preemptive_future[request.label]:
            self.update_user_fact(user, fact, date, request.label)
            # NOTE preemptive did not finish before user responded
            # mark commit as taken
            self.preemptive_commit[user.user_id] = 'done'
        else:
            results = self.preemptive_future[request.label][user.user_id]
            self.preemptive_commit[user.user_id] = results
            user = results['user']
            fact = results['fact']

        if user.user_id in self.debug_id:
            self.debug_id.pop(user.user_id)

        # self.db.update_user(user)
        # self.db.update_fact(fact)
        # self.db.commit()

        # update user stats
        deck_id = 'all' if not fact.deck_id else fact.deck_id
        self.update_user_stats(session, user, record, deck_id=deck_id)

        session.commit()

        detail = {
            # 'response': request.label,
            # 'old ltn box': old_user.leitner_box.get(fact.fact_id, '-'),
            # 'new ltn box': user.leitner_box.get(fact.fact_id, '-'),
            # 'old ltn dat': str(old_user.leitner_scheduled_date.get(fact.fact_id, '-')),
            # 'new ltn dat': str(user.leitner_scheduled_date.get(fact.fact_id, '-')),
            # 'old sm2 rep': old_user.sm2_repetition.get(fact.fact_id, '-'),
            # 'new sm2 rep': user.sm2_repetition.get(fact.fact_id, '-'),
            # 'old sm2 inv': old_user.sm2_interval.get(fact.fact_id, '-'),
            # 'new sm2 inv': user.sm2_interval.get(fact.fact_id, '-'),
            # 'old sm2 e_f': old_user.sm2_efactor.get(fact.fact_id, '-'),
            # 'new sm2 e_f': user.sm2_efactor.get(fact.fact_id, '-'),
            # 'old sm2 dat': str(old_user.sm2_scheduled_date.get(fact.fact_id, '-')),
            # 'new sm2 dat': str(user.sm2_scheduled_date.get(fact.fact_id, '-')),
        }

        return detail

    def update_user_stats(self, session, user: User, record: Record, deck_id: str):
        curr_stat = session.query(UserStat).\
            filter(UserStat.user_id == user.user_id).\
            filter(UserStat.deck_id == deck_id).\
            order_by(UserStat.date.desc()).first()

        is_new_stat = False
        if curr_stat is None:
            user_stat_id = json.dumps({
                'user_id': user.user_id,
                'date': str(record.date.date()),
                'deck_id': deck_id,
            })
            curr_stat = UserStat(
                user_stat_id=user_stat_id,
                user_id=user.user_id,
                deck_id=deck_id,
                date=record.date.date(),
                new_facts=0,
                reviewed_facts=0,
                new_correct=0,
                reviewed_correct=0,
                total_seen=0,
                total_milliseconds=0,
                total_seconds=0,
                total_minutes=0,
                elapsed_milliseconds_text=0,
                elapsed_milliseconds_answer=0,
                elapsed_seconds_text=0,
                elapsed_seconds_answer=0,
                elapsed_minutes_text=0,
                elapsed_minutes_answer=0,
            )
            is_new_stat = True

        if record.date.date() != curr_stat.date:
            # there is a previous user_stat, but not from today
            # copy user stat to today
            user_stat_id = json.dumps({
                'user_id': user.user_id,
                'date': str(record.date.date()),
                'deck_id': deck_id,
            })
            new_stat = UserStat(
                user_stat_id=user_stat_id,
                user_id=user.user_id,
                deck_id=deck_id,
                date=record.date.date(),
                new_facts=curr_stat.new_facts,
                reviewed_facts=curr_stat.reviewed_facts,
                new_correct=curr_stat.new_correct,
                reviewed_correct=curr_stat.reviewed_correct,
                total_seen=curr_stat.total_seen,
                total_milliseconds=curr_stat.total_milliseconds,
                total_seconds=curr_stat.total_seconds,
                total_minutes=curr_stat.total_minutes,
                elapsed_milliseconds_text=curr_stat.elapsed_milliseconds_text,
                elapsed_milliseconds_answer=curr_stat.elapsed_milliseconds_answer,
                elapsed_seconds_text=curr_stat.elapsed_seconds_text,
                elapsed_seconds_answer=curr_stat.elapsed_seconds_answer,
                elapsed_minutes_text=curr_stat.elapsed_minutes_text,
                elapsed_minutes_answer=curr_stat.elapsed_minutes_answer,
            )
            curr_stat = new_stat
            is_new_stat = True

        if record.is_new_fact:
            curr_stat.new_facts += 1
            curr_stat.new_correct += record.response
        else:
            curr_stat.reviewed_facts += 1
            curr_stat.reviewed_correct += record.response

        total_milliseconds = record.elapsed_milliseconds_text + record.elapsed_milliseconds_answer
        curr_stat.total_seen += 1
        curr_stat.total_milliseconds += total_milliseconds
        curr_stat.total_seconds += total_milliseconds // 1000
        curr_stat.total_minutes += total_milliseconds // 60000
        curr_stat.elapsed_milliseconds_text += record.elapsed_milliseconds_text
        curr_stat.elapsed_milliseconds_answer += record.elapsed_milliseconds_answer
        curr_stat.elapsed_seconds_text += record.elapsed_milliseconds_text // 1000
        curr_stat.elapsed_seconds_answer += record.elapsed_milliseconds_answer // 1000
        curr_stat.elapsed_minutes_text += record.elapsed_milliseconds_text // 60000
        curr_stat.elapsed_minutes_answer += record.elapsed_milliseconds_answer // 60000

        if is_new_stat:
            session.add(curr_stat)

    def leitner_update(self, user: User, fact: Fact, response: bool) -> None:
        """
        Update Leitner box and scheduled date of card.

        :param user:
        :param fact:
        :param response: CORRECT or WRONG.
        """
        # TODO handle correct on first try

        # leitner boxes 1~10
        # days[0] = None as placeholder since we don't have box 0
        # days[9] and days[10] = 9999 to make it never repeat
        days = [0, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 9999, 9999]
        increment_days = {i: x for i, x in enumerate(days)}

        # boxes: 1 ~ 10
        cur_box = user.leitner_box.get(fact.fact_id, None)
        if cur_box is None:
            cur_box = 1
        new_box = cur_box + (1 if response == CORRECT else -1)
        new_box = max(min(new_box, 10), 1)
        user.leitner_box[fact.fact_id] = new_box
        interval = timedelta(days=increment_days[new_box])
        # NOTE we increment on top `previous_study`, so it should be updated in
        # `update` before `leitner_update` is called.
        # it should correpond to the latest study date.
        prev_date, prev_response = user.previous_study[fact.fact_id]
        if isinstance(prev_date, str):
            prev_date = parse_date(prev_date)
        user.leitner_scheduled_date[fact.fact_id] = str(prev_date + interval)

    def sm2_update(self, user: User, fact: Fact, response: bool) -> None:
        """
        Update SM-2 e_factor, repetition, and interval.


        :param user:
        :param fact:
        :param response:
        """
        def get_quality_from_response(response: bool) -> int:
            return 4 if response == CORRECT else 1

        # TODO handle correct on first try

        e_f = user.sm2_efactor.get(fact.fact_id, 2.5)
        inv = user.sm2_interval.get(fact.fact_id, 1)
        rep = user.sm2_repetition.get(fact.fact_id, 0) + 1

        q = get_quality_from_response(response)
        e_f = max(1.3, e_f + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

        if response != CORRECT:
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
        if isinstance(prev_date, str):
            prev_date = parse_date(prev_date)
        user.sm2_scheduled_date[fact.fact_id] = str(prev_date + timedelta(days=inv))

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
