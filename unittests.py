import os
import json
import copy
import pickle
import unittest
import numpy as np
from datetime import datetime

from db import SchedulerDB
from util import User, Card, History, Params
from scheduler import MovingAvgScheduler


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
            last_study_date={'card 1': datetime.now()},
            leitner_box={'card 1': 2},
            leitner_scheduled_date={'card 2': datetime.now()},
            sm2_efactor={'card 1': 0.5},
            sm2_interval={'card 1': 6},
            sm2_repetition={'card 1': 10},
            sm2_scheduled_date={'card 2': datetime.now()},
            date=datetime.now()
        )
        self.assertFalse(self.db.check_user(user.user_id))
        self.assertFalse(self.db.get_user(user.user_id))
        self.db.add_user(user)
        self.assertTrue(self.db.check_user(user.user_id))
        returned_user = self.db.get_user(user.user_id)
        self.assert_user_equal(user, returned_user)

        user.qrep.append(np.array([0.7, 0.8, 0.9]))
        user.skill.append(np.array([0.4, 0.5, 0.6]))
        user.date = datetime.now()
        self.db.update_user(user)
        returned_user = self.db.get_user(user.user_id)
        self.assert_user_equal(user, returned_user)

    def assert_user_equal(self, u1, u2):
        self.assertEqual(u1.user_id, u2.user_id)
        np.testing.assert_array_equal(u1.qrep, u2.qrep)
        np.testing.assert_array_equal(u1.skill, u2.skill)
        self.assertEqual(u1.last_study_date, u2.last_study_date)
        self.assertEqual(u1.leitner_box, u2.leitner_box)
        self.assertEqual(u1.leitner_scheduled_date, u2.leitner_scheduled_date)
        self.assertEqual(u1.sm2_efactor, u2.sm2_efactor)
        self.assertEqual(u1.sm2_interval, u2.sm2_interval)
        self.assertEqual(u1.sm2_repetition, u2.sm2_repetition)
        self.assertEqual(u1.sm2_scheduled_date, u2.sm2_scheduled_date)
        self.assertEqual(u1.date, u2.date)

    def assert_card_equal(self, c1, c2):
        self.assertEqual(c1.card_id, c2.card_id)
        self.assertEqual(c1.text, c2.text)
        self.assertEqual(c1.answer, c2.answer)
        np.testing.assert_array_equal(c1.qrep, c2.qrep)
        np.testing.assert_array_equal(c1.skill, c2.skill)
        self.assertEqual(c1.category, c2.category)
        self.assertEqual(c1.date, c2.date)

    def test_card(self):
        card = Card(
            card_id='card 1',
            text='This is the question text',
            answer='Answer Text III',
            qrep=np.array([1, 2, 3, 4]),
            skill=np.array([0.1, 0.2, 0.3, 0.4]),
            category='WORLD',
            date=datetime.now()
        )

        self.assertFalse(self.db.check_card(card.card_id))
        self.db.add_card(card)
        self.assertTrue(self.db.check_card(card.card_id))
        returned_card = self.db.get_card(card.card_id)
        self.assert_card_equal(card, returned_card)

        card.__dict__.update({
            'text': 'This is the NEWWWWWWW question text',
            'answer': 'Answer Text IVVVV',
            'qrep': np.array([1, 2, 3, 4]),
            'skill': np.array([0.1, 0.7, 0.3, 0.8]),
            'category': 'WORLD',
            'date': datetime.now()
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
            last_study_date={'card 1': datetime.now()},
            leitner_box={'card 1': 2},
            leitner_scheduled_date={'card 2': datetime.now()},
            sm2_efactor={'card 1': 0.5},
            sm2_interval={'card 1': 6},
            sm2_repetition={'card 1': 10},
            sm2_scheduled_date={'card 2': datetime.now()},
            date=datetime.now()
        )
        card = Card(
            card_id='card 1',
            text='This is the question text',
            answer='Answer Text III',
            qrep=np.array([1, 2, 3, 4]),
            skill=np.array([0.1, 0.2, 0.3, 0.4]),
            category='WORLD',
            date=datetime.now()
        )
        params = Params()
        old_history_id = json.dumps({'user_id': user.user_id, 'card_id': card.card_id})
        history = History(
            history_id=old_history_id,
            user_id=user.user_id,
            card_id=card.card_id,
            response='User Guess',
            judgement='wrong',
            user_snapshot=user.to_snapshot(),
            scheduler_snapshot=json.dumps(params.__dict__),
            card_ids=json.dumps([1, 2, 3, 4, 5]),
            scheduler_output='(awd, awd, awd)',
            date=datetime.now())
        self.db.add_history(history)
        returned_history = self.db.get_history(old_history_id)
        returned_user = User.from_snapshot(returned_history.user_snapshot)
        self.assert_user_equal(user, returned_user)
        new_history_id = 'real_history_id'
        history.__dict__.update({
            'history_id': new_history_id,
            'date': datetime.now()
        })
        self.db.update_history(old_history_id, history)
        returned_history = self.db.get_history(new_history_id)
        returned_user = User.from_snapshot(returned_history.user_snapshot)
        self.assertEqual(returned_history.history_id, new_history_id)
        self.assert_user_equal(user, returned_user)
        self.assertFalse(self.db.check_history(old_history_id))


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
        self.assertEqual(u1.last_study_date, u2.last_study_date)
        self.assertEqual(u1.leitner_box, u2.leitner_box)
        self.assertEqual(u1.leitner_scheduled_date, u2.leitner_scheduled_date)
        self.assertEqual(u1.sm2_efactor, u2.sm2_efactor)
        self.assertEqual(u1.sm2_interval, u2.sm2_interval)
        self.assertEqual(u1.sm2_repetition, u2.sm2_repetition)
        self.assertEqual(u1.sm2_scheduled_date, u2.sm2_scheduled_date)
        self.assertEqual(u1.date, u2.date)

    def assert_card_equal(self, c1, c2):
        self.assertEqual(c1.card_id, c2.card_id)
        self.assertEqual(c1.text, c2.text)
        self.assertEqual(c1.answer, c2.answer)
        np.testing.assert_array_equal(c1.qrep, c2.qrep)
        np.testing.assert_array_equal(c1.skill, c2.skill)
        self.assertEqual(c1.category, c2.category)
        self.assertEqual(c1.date, c2.date)

    def test_scheduler_update(self):
        with open('data/diagnostic_questions.pkl', 'rb') as f:
            cards = pickle.load(f)
        cards = cards[:5]
        for i, c in enumerate(cards):
            cards[i]['user_id'] = 'shi'
            cards[i]['date'] = str(datetime.now())

        print(cards[0])

        # using deepcopy here because DB lookup converts dict cards into Card
        # cards. not a problem in actual use because objects go through web API
        # and not reused
        result = self.scheduler.schedule(copy.deepcopy(cards))
        order = result['order']

        print()
        print()
        print(cards[0])

        card_selected = cards[order[0]]
        card_selected.update({
            'label': 'correct',
            'history_id': 'real_history_id'
        })

        self.scheduler.update([card_selected])

        h = self.scheduler.db.get_history()[0]
        for key, value in h.__dict__.items():
            print(key, value)

        print()
        print()
        user = self.scheduler.db.get_user('shi')
        print(user)


if __name__ == '__main__':
    unittest.main()
