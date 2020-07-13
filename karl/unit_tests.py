#!/usr/bin/env python
# coding: utf-8

import os
import json
import copy
import pickle
import unittest
import requests
import numpy as np
from pprint import pprint
from datetime import datetime, timedelta
from dateutil.parser import parse as parse_date

from karl.db import SchedulerDB
from karl.util import User, Fact, History, Params, ScheduleRequest
from karl.scheduler import MovingAvgScheduler


CORRECT = True
WRONG = False


class TestDB(unittest.TestCase):

    def setUp(self):
        self.filename = 'db_test.sqlite'
        if os.path.exists(self.filename):
            os.remove(self.filename)
        self.db = SchedulerDB(self.filename)

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def test_user(self):
        fact = Fact(
            fact_id='fact 1',
            text='This is the question text',
            answer='Answer Text III',
            category='WORLD',
            deck_name='deck_name_test',
            deck_id='deck_id_test',
            qrep=np.array([1, 2, 3, 4]),
            skill=np.array([0.1, 0.2, 0.3, 0.4]),
            results=[True, False, True, True]
        )
        user = User(
            user_id='user 1',
            recent_facts=[fact],
            previous_study={'fact 1': (datetime.now(), CORRECT)},
            leitner_box={'fact 1': 2},
            leitner_scheduled_date={'fact 2': datetime.now()},
            sm2_efactor={'fact 1': 0.5},
            sm2_interval={'fact 1': 6},
            sm2_repetition={'fact 1': 10},
            sm2_scheduled_date={'fact 2': datetime.now()},
            results=[True, False, True],
            count_correct_before={'fact 1': 1},
            count_wrong_before={'fact 1': 3}
        )
        self.assertFalse(self.db.check_user(user.user_id))
        self.assertFalse(self.db.get_user(user.user_id))
        self.db.add_user(user)
        self.assertTrue(self.db.check_user(user.user_id))
        returned_user = self.db.get_user(user.user_id)
        self.assert_user_equal(user, returned_user)

        user.results.append(False)
        user.count_correct_before['fact 1'] = 2
        user.date = datetime.now()
        self.db.update_user(user)
        returned_user = self.db.get_user(user.user_id)
        self.assert_user_equal(user, returned_user)

    def assert_user_equal(self, u1, u2):
        self.assertEqual(u1.user_id, u2.user_id)
        self.assert_fact_equal(u1.recent_facts[0], u2.recent_facts[0])
        self.assertEqual(u1.previous_study, u2.previous_study)
        self.assertEqual(u1.leitner_box, u2.leitner_box)
        self.assertEqual(u1.leitner_scheduled_date, u2.leitner_scheduled_date)
        self.assertEqual(u1.sm2_efactor, u2.sm2_efactor)
        self.assertEqual(u1.sm2_interval, u2.sm2_interval)
        self.assertEqual(u1.sm2_repetition, u2.sm2_repetition)
        self.assertEqual(u1.sm2_scheduled_date, u2.sm2_scheduled_date)
        self.assertEqual(u1.results, u2.results)
        self.assertEqual(u1.count_correct_before, u2.count_correct_before)
        self.assertEqual(u1.count_wrong_before, u2.count_wrong_before)
        self.assertEqual(u1.params, u2.params)

    def assert_fact_equal(self, c1, c2):
        self.assertEqual(c1.fact_id, c2.fact_id)
        self.assertEqual(c1.text, c2.text)
        self.assertEqual(c1.answer, c2.answer)
        self.assertEqual(c1.category, c2.category)
        self.assertEqual(c1.deck_name, c2.deck_name)
        self.assertEqual(c1.deck_id, c2.deck_id)
        np.testing.assert_array_equal(c1.qrep, c2.qrep)
        np.testing.assert_array_equal(c1.skill, c2.skill)
        self.assertEqual(c1.results, c2.results)

    def test_fact(self):
        fact = Fact(
            fact_id='fact 1',
            text='This is the question text',
            answer='Answer Text III',
            category='WORLD',
            deck_name='deck_name_test',
            deck_id='deck_id_test',
            qrep=np.array([1, 2, 3, 4]),
            skill=np.array([0.1, 0.2, 0.3, 0.4]),
            results=[True, False, True, True]
        )

        self.assertFalse(self.db.check_fact(fact.fact_id))
        self.db.add_fact(fact)
        self.assertTrue(self.db.check_fact(fact.fact_id))
        returned_fact = self.db.get_fact(fact.fact_id)
        self.assert_fact_equal(fact, returned_fact)

        fact.__dict__.update({
            'text': 'This is the NEWWWWWWW question text',
            'answer': 'Answer Text IVVVV',
            'category': 'WORLD',
            'qrep': np.array([1, 2, 3, 4]),
            'skill': np.array([0.1, 0.7, 0.3, 0.8]),
            'results': [True, False, True, True, False]
        })
        self.db.update_fact(fact)
        returned_fact = self.db.get_fact(fact.fact_id)
        self.assert_fact_equal(fact, returned_fact)

    def test_history(self):
        fact = Fact(
            fact_id='fact 1',
            text='This is the question text',
            answer='Answer Text III',
            category='WORLD',
            deck_name='deck_name_test',
            deck_id='deck_id_test',
            qrep=np.array([1, 2, 3, 4]),
            skill=np.array([0.1, 0.2, 0.3, 0.4]),
            results=[True, False, True, True]
        )
        user = User(
            user_id='user 1',
            recent_facts=[fact],
            previous_study={'fact 1': (datetime.now(), CORRECT)},
            leitner_box={'fact 1': 2},
            leitner_scheduled_date={'fact 2': datetime.now()},
            sm2_efactor={'fact 1': 0.5},
            sm2_interval={'fact 1': 6},
            sm2_repetition={'fact 1': 10},
            sm2_scheduled_date={'fact 2': datetime.now()},
            results=[True, False, True],
            count_correct_before={'fact 1': 1},
            count_wrong_before={'fact 1': 3}
        )
        params = Params()
        old_history_id = json.dumps({'user_id': user.user_id, 'fact_id': fact.fact_id})
        history = History(
            history_id=old_history_id,
            debug_id='random_debug_id',
            user_id=user.user_id,
            fact_id=fact.fact_id,
            deck_id=fact.deck_id,
            response='User Guess',
            judgement=WRONG,
            user_snapshot=json.dumps(user.pack()),
            scheduler_snapshot=json.dumps(params.__dict__),
            fact_ids=json.dumps([1, 2, 3, 4, 5]),
            scheduler_output='(awd, awd, awd)',
            date=datetime.now())
        self.db.add_history(history)
        returned_history = self.db.get_history(old_history_id)
        returned_user = User.unpack(returned_history.user_snapshot)
        self.assert_user_equal(user, returned_user)
        new_history_id = 'real_history_id'
        history.__dict__.update({
            'history_id': new_history_id,
            'date': datetime.now()
        })
        self.db.update_history(old_history_id, history)
        returned_history = self.db.get_history(new_history_id)
        returned_user = User.unpack(returned_history.user_snapshot)
        self.assertEqual(returned_history.history_id, new_history_id)
        self.assert_user_equal(user, returned_user)
        self.assertFalse(self.db.check_history(old_history_id))

        new_user = copy.deepcopy(user)
        new_user.user_id = 'new user 1'
        new_user.sm2_efactor = {'fact 1': 0.05},
        self.assert_user_equal(user, returned_user)


class TestScheduler(unittest.TestCase):

    def setUp(self):
        self.filename_w_pre = 'db_test_w_pre.sqlite'
        if os.path.exists(self.filename_w_pre):
            os.remove(self.filename_w_pre)
        self.scheduler_w = MovingAvgScheduler(
            db_filename=self.filename_w_pre, preemptive=True)

        self.filename_wo_pre = 'db_test_wo_pre.sqlite'
        if os.path.exists(self.filename_wo_pre):
            os.remove(self.filename_wo_pre)
        self.scheduler_wo = MovingAvgScheduler(
            db_filename=self.filename_wo_pre, preemptive=False)

    def tearDown(self):
        if os.path.exists(self.filename_w_pre):
            os.remove(self.filename_w_pre)
        if os.path.exists(self.filename_wo_pre):
            os.remove(self.filename_wo_pre)

    def assert_user_equal(self, u1, u2):
        self.assertEqual(u1.user_id, u2.user_id)
        self.assert_fact_equal(u1.recent_facts[0], u2.recent_facts[0])
        self.assertEqual(u1.previous_study_date, u2.previous_study_date)
        self.assertEqual(u1.leitner_box, u2.leitner_box)
        self.assertEqual(u1.leitner_scheduled_date, u2.leitner_scheduled_date)
        self.assertEqual(u1.sm2_efactor, u2.sm2_efactor)
        self.assertEqual(u1.sm2_interval, u2.sm2_interval)
        self.assertEqual(u1.sm2_repetition, u2.sm2_repetition)
        self.assertEqual(u1.sm2_scheduled_date, u2.sm2_scheduled_date)
        self.assertEqual(u1.results, u2.results)
        self.assertEqual(u1.count_correct_before, u2.count_correct_before)
        self.assertEqual(u1.count_wrong_before, u2.count_wrong_before)
        self.assertEqual(u1.params, u2.params)

    def assert_fact_equal(self, c1, c2):
        self.assertEqual(c1.fact_id, c2.fact_id)
        self.assertEqual(c1.text, c2.text)
        self.assertEqual(c1.answer, c2.answer)
        self.assertEqual(c1.category, c2.category)
        np.testing.assert_array_equal(c1.qrep, c2.qrep)
        np.testing.assert_array_equal(c1.skill, c2.skill)
        self.assertEqual(c1.results, c2.results)

    def test_scheduler_update(self):
        with open('data/diagnostic_questions.pkl', 'rb') as f:
            facts = pickle.load(f)
        facts = facts[:5]

        user_id = 'test_dummy'
        self.scheduler_w.reset_user(user_id)
        self.scheduler_wo.reset_user(user_id)
        requests = []
        for c in facts:
            self.scheduler_w.reset_fact(c['fact_id'])
            self.scheduler_wo.reset_fact(c['fact_id'])
            requests.append(
                ScheduleRequest(
                    user_id=user_id,
                    fact_id=c['fact_id'],
                    text=c['text'],
                    answer=c['answer'],
                    category=c['category'],
                    deck_name='deck_name_test',
                    deck_id='deck_id_test',
                )
            )

        print()

        start_date = parse_date('2028-06-1 08:00:00.000001')
        for i in range(5):
            current_date = start_date + timedelta(seconds=i * 30)
            for r in requests:
                r.date = current_date

            results_w = self.scheduler_w.schedule(copy.deepcopy(requests), current_date)
            results_wo = self.scheduler_wo.schedule(copy.deepcopy(requests), current_date)
            print(i)
            print('scores_p', [s['sum'] for s in results_w['scores']])
            print('scores_o', [s['sum'] for s in results_wo['scores']])

            self.assertEqual(results_w['order'], results_wo['order'])
            order = results_w['order']
            request = requests[order[0]]
            request.__dict__.update({
                'label': CORRECT,
                'history_id': 'real_history_id_{}_{}'.format(user_id, request.fact_id)
            })
            self.scheduler_w.update([request], current_date)
            self.scheduler_wo.update([request], current_date)

            if user_id not in self.scheduler_w.preemptive_commit:
                print('not in commit')
                print(self.scheduler_w.preemptive_future.keys())
                print(self.scheduler_w.preemptive_future[CORRECT].keys())
                print(self.scheduler_w.preemptive_future['wrong'].keys())
            elif self.scheduler_w.preemptive_commit[user_id] != 'done':
                facts_w = self.scheduler_w.preemptive_commit[user_id]['facts']
                print('facts_wp', [c.results for c in facts_w])

            facts_wo = self.scheduler_wo.get_facts(requests)
            print('facts_wo', [c.results for c in facts_wo])

            if user_id not in self.scheduler_w.preemptive_commit:
                pass
            elif self.scheduler_w.preemptive_commit[user_id] != 'done':
                user_w = self.scheduler_w.preemptive_commit[user_id]['user']
                print('users_wp', user_w.results)
            user_wo = self.scheduler_wo.get_user(user_id)
            print('users_wo', user_wo.results)
            print()


class TestWeb(unittest.TestCase):

    def test_user_stats(self):
        env = 'dev'
        user_id = '2580'
        URL = 'http://127.0.0.1:8000/api/karl'
        deck_id = 'unit_test_deck'

        date_start = '2028-06-03T03:41:14.779779-0400'

        with open('data/diagnostic_questions.pkl', 'rb') as f:
            diagnostic_facts = pickle.load(f)

        facts = copy.deepcopy(diagnostic_facts[:10])
        for fact in facts:
            fact.update({
                'env': env,
                'deck_id': deck_id,
                'user_id': user_id,
            })

        # prepare simulated user
        requests.get(f'{URL}/reset_user?user_id={user_id}&env={env}')

        for day in range(10):
            for i in range(10):
                date = parse_date(date_start) + timedelta(days=day, seconds=i)

                fact = facts[i]
                fact['label'] = True
                fact['date'] = date.strftime('%Y-%m-%dT%H:%M:%S%z')

                # update scheduler with binary outcome
                fact['history_id'] = f'{user_id}_{fact["fact_id"]}_{fact["date"]}'
                fact['elapsed_seconds_text'] = 2
                fact['elapsed_seconds_answer'] = 2
                requests.post(f'{URL}/update', data=json.dumps([fact]))

        # req = f'{URL}/get_user_stats?user_id={user_id}&env={env}&date_start={date_start}'
        # print(req)
        # stats = json.loads(requests.get(req).text)
        # pprint(stats)
        # print()
        # print()

        date_end = parse_date(date_start) + timedelta(days=1)
        date_end = date_end.strftime('%Y-%m-%dT%H:%M:%S%z')
        min_studied = 10
        r = requests.get(f'{URL}/leaderboard?env={env}&min_studied={min_studied}&date_start={date_start}&date_end={date_end}')
        leaderboard = json.loads(r.text)
        pprint(leaderboard)


if __name__ == '__main__':
    unittest.main()
