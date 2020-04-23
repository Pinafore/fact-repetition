import os
import json
import copy
import pickle
import unittest
import numpy as np
from datetime import datetime, timedelta

from karl.db import SchedulerDB
from karl.util import User, Card, History, Params, ScheduleRequest, parse_date
from karl.scheduler import MovingAvgScheduler


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
        user = User(
            user_id='user 1',
            qrep=[np.array([0.1, 0.2, 0.3])],
            skill=[np.array([0.1, 0.2, 0.3])],
            category='History',
            previous_study={'card 1': (datetime.now(), 'correct')},
            leitner_box={'card 1': 2},
            leitner_scheduled_date={'card 2': datetime.now()},
            sm2_efactor={'card 1': 0.5},
            sm2_interval={'card 1': 6},
            sm2_repetition={'card 1': 10},
            sm2_scheduled_date={'card 2': datetime.now()},
            results=[True, False, True],
            count_correct_before={'card 1': 1},
            count_wrong_before={'card 1': 3}
        )
        self.assertFalse(self.db.check_user(user.user_id))
        self.assertFalse(self.db.get_user(user.user_id))
        self.db.add_user(user)
        self.assertTrue(self.db.check_user(user.user_id))
        returned_user = self.db.get_user(user.user_id)
        self.assert_user_equal(user, returned_user)

        user.qrep.append(np.array([0.7, 0.8, 0.9]))
        user.skill.append(np.array([0.4, 0.5, 0.6]))
        user.results.append(False)
        user.count_correct_before['card 1'] = 2
        user.date = datetime.now()
        self.db.update_user(user)
        returned_user = self.db.get_user(user.user_id)
        self.assert_user_equal(user, returned_user)

    def assert_user_equal(self, u1, u2):
        self.assertEqual(u1.user_id, u2.user_id)
        np.testing.assert_array_equal(u1.qrep, u2.qrep)
        np.testing.assert_array_equal(u1.skill, u2.skill)
        self.assertEqual(u1.category, u2.category)
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

    def assert_card_equal(self, c1, c2):
        self.assertEqual(c1.card_id, c2.card_id)
        self.assertEqual(c1.text, c2.text)
        self.assertEqual(c1.answer, c2.answer)
        self.assertEqual(c1.category, c2.category)
        np.testing.assert_array_equal(c1.qrep, c2.qrep)
        np.testing.assert_array_equal(c1.skill, c2.skill)
        self.assertEqual(c1.results, c2.results)

    def test_card(self):
        card = Card(
            card_id='card 1',
            text='This is the question text',
            answer='Answer Text III',
            category='WORLD',
            qrep=np.array([1, 2, 3, 4]),
            skill=np.array([0.1, 0.2, 0.3, 0.4]),
            results=[True, False, True, True]
        )

        self.assertFalse(self.db.check_card(card.card_id))
        self.db.add_card(card)
        self.assertTrue(self.db.check_card(card.card_id))
        returned_card = self.db.get_card(card.card_id)
        self.assert_card_equal(card, returned_card)

        card.__dict__.update({
            'text': 'This is the NEWWWWWWW question text',
            'answer': 'Answer Text IVVVV',
            'category': 'WORLD',
            'qrep': np.array([1, 2, 3, 4]),
            'skill': np.array([0.1, 0.7, 0.3, 0.8]),
            'results': [True, False, True, True, False]
        })
        self.db.update_card(card)
        returned_card = self.db.get_card(card.card_id)
        self.assert_card_equal(card, returned_card)

    def test_history(self):
        user = User(
            user_id='user 1',
            qrep=[np.array([0.1, 0.2, 0.3])],
            skill=[np.array([0.1, 0.2, 0.3])],
            category='History',
            previous_study={'card 1': (datetime.now(), 'correct')},
            leitner_box={'card 1': 2},
            leitner_scheduled_date={'card 2': datetime.now()},
            sm2_efactor={'card 1': 0.5},
            sm2_interval={'card 1': 6},
            sm2_repetition={'card 1': 10},
            sm2_scheduled_date={'card 2': datetime.now()},
            results=[True, False, True],
            count_correct_before={'card 1': 1},
            count_wrong_before={'card 1': 3}
        )
        card = Card(
            card_id='card 1',
            text='This is the question text',
            answer='Answer Text III',
            category='WORLD',
            qrep=np.array([1, 2, 3, 4]),
            skill=np.array([0.1, 0.2, 0.3, 0.4]),
            results=[True, False, True, True]
        )
        params = Params()
        old_history_id = json.dumps({'user_id': user.user_id, 'card_id': card.card_id})
        history = History(
            history_id=old_history_id,
            user_id=user.user_id,
            card_id=card.card_id,
            response='User Guess',
            judgement='wrong',
            user_snapshot=json.dumps(user.pack()),
            scheduler_snapshot=json.dumps(params.__dict__),
            card_ids=json.dumps([1, 2, 3, 4, 5]),
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

        t0 = datetime.now()
        new_user = copy.deepcopy(user)
        new_user.user_id = 'new user 1'
        new_user.sm2_efactor = {'card 1': 0.05},
        self.assert_user_equal(user, returned_user)
        print('deep copy user takes', datetime.now() - t0)


class TestScheduler(unittest.TestCase):

    def setUp(self):
        self.filename = 'db_test.sqlite'
        if os.path.exists(self.filename):
            os.remove(self.filename)
        self.scheduler = MovingAvgScheduler(db_filename=self.filename)
        self.db = self.scheduler.db

    def tearDown(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

    def assert_user_equal(self, u1, u2):
        self.assertEqual(u1.user_id, u2.user_id)
        np.testing.assert_array_equal(u1.qrep, u2.qrep)
        np.testing.assert_array_equal(u1.skill, u2.skill)
        self.assertEqual(u1.category, u2.category)
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

    def assert_card_equal(self, c1, c2):
        self.assertEqual(c1.card_id, c2.card_id)
        self.assertEqual(c1.text, c2.text)
        self.assertEqual(c1.answer, c2.answer)
        self.assertEqual(c1.category, c2.category)
        np.testing.assert_array_equal(c1.qrep, c2.qrep)
        np.testing.assert_array_equal(c1.skill, c2.skill)
        self.assertEqual(c1.results, c2.results)

    def test_scheduler_update(self):
        with open('data/diagnostic_questions.pkl', 'rb') as f:
            cards = pickle.load(f)
        cards = cards[:5]

        scores = []

        for user_id, params in zip(
                ['precompute_no', 'precompute_yes'], 
                [Params(precompute=False), Params(precompute=True)]
        ):
            self.scheduler.set_params(params)
            self.scheduler.reset_user(user_id)
            
            requests = []
            for c in cards:
                self.scheduler.reset_card(c['question_id'])
                requests.append(
                    ScheduleRequest(
                        user_id=user_id,
                        text=c['text'],
                        answer=c['answer'],
                        category=c['category'],
                        question_id=c['question_id']
                    )
                )

            scores.append([])
            start_date = parse_date('2028-06-1 08:00:00.000001')
            print()
            for i in range(5):
                current_date = start_date + timedelta(seconds=i * 30)
                for r in requests:
                    r.date = current_date

                if user_id in self.scheduler.precompute_commit:
                    returned_cards = self.scheduler.precompute_commit[user_id]['cards']
                    print(i, 'yes card', [c.results for c in returned_cards])
                else:
                    returned_cards = self.scheduler.get_cards(requests)
                    print(i, 'no  card', [c.results for c in returned_cards])

                if user_id in self.scheduler.precompute_commit:
                    returned_user = self.scheduler.precompute_commit[user_id]['user']
                    print(i, 'yes user', returned_user.results)
                else:
                    returned_user = self.scheduler.get_user(user_id)
                    print(i, 'no  user', returned_user.results)

                results = self.scheduler.schedule(copy.deepcopy(requests), datetime.now())
                order = results['order']
                request = requests[order[0]]
                request.__dict__.update({
                    'label': 'correct',
                    'history_id': 'real_history_id_{}_{}'.format(user_id, request.question_id)
                })
                self.scheduler.update([request], datetime.now())
                scores[-1].append([s['sum'] for s in results['scores']])

        print()
        for ss in list(zip(*scores)):
            print(ss[0])
            print(ss[1])
            print()


if __name__ == '__main__':
    unittest.main()
