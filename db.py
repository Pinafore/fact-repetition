import os
import json
import sqlite3
import logging
import numpy as np
from datetime import datetime
from util import User, Card, History, Params
from typing import Optional

logger = logging.getLogger('scheduler')
DB_FILENAME = 'db.sqlite'

class SchedulerDB:

    def __init__(self):
        if not os.path.exists(DB_FILENAME):
            self.create()
        self.conn = sqlite3.connect(DB_FILENAME)

    def create(self):
        conn = sqlite3.connect(DB_FILENAME)
        c = conn.cursor()

        # *current* state of the player
        # including leitner and sm2 info
        c.execute('CREATE TABLE users (\
                   user_id PRIMARY KEY, \
                   qrep TEXT, \
                   skill TEXT, \
                   repetition TEXT, \
                   last_study_time TEXT, \
                   scheduled_time TEXT, \
                   sm2_efactor TEXT, \
                   sm2_interval TEXT, \
                   leitner_box TEXT, \
                   last_update timestamp)')

        # *current* cache of cards
        c.execute('CREATE TABLE cards (\
                   card_id PRIMARY KEY, \
                   text TEXT, \
                   answer TEXT, \
                   qrep TEXT, \
                   skill TEXT, \
                   category TEXT, \
                   last_update timestamp)')

        # input to and output from the schedule API
        c.execute('CREATE TABLE history (\
                   history_id PRIMARY KEY, \
                   user_id TEXT, \
                   card_id TEXT, \
                   response TEXT, \
                   judgement TEXT, \
                   user_snapshot TEXT, \
                   scheduler_snapshot TEXT, \
                   cards TEXT, \
                   scheduler_output TEXT, \
                   timestamp timestamp)')

        conn.commit()
        conn.close()

    def add_user(self, u: User):
        c = self.conn.cursor()
        try:
            c.execute('INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?)',
                      (
                          u.user_id,
                          json.dumps(u.qrep.tolist()),
                          json.dumps(u.skill),
                          json.dumps(u.repetition),
                          json.dumps({k: str(v) for k, v in u.last_study_time.items()}),
                          json.dumps({k: str(v) for k, v in u.scheduled_time.items()}),
                          json.dumps(u.sm2_efactor),
                          json.dumps(u.sm2_interval),
                          json.dumps(u.leitner_box),
                          u.last_update))
        except sqlite3.IntegrityError:
            logger.info("user {} exists".format(u.user_id))
            print("user {} exists".format(u.user_id))
        self.conn.commit()

    def get_user(self, user_id: str = None):
        def row_to_dict(r):
            return User(
                user_id=r[0],
                qrep=np.array(json.loads(r[1])),
                skill=json.loads(r[2]),
                repetition=json.loads(r[3]),
                last_study_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                                 for k, v in json.loads(r[4]).items()},
                scheduled_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                                for k, v in json.loads(r[5]).items()},
                sm2_efactor=json.loads(r[6]),
                sm2_interval=json.loads(r[7]),
                leitner_box=json.loads(r[8]),
                last_update=r[9])
        c = self.conn.cursor()
        if user_id is None:
            c.execute("SELECT * FROM users")
            return [row_to_dict(r) for r in c.fetchall()]
        else:
            c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
            r = c.fetchone()
            return row_to_dict(r) if r else None

    def check_user(self, user_id: str) -> bool:
        c = self.conn.cursor()
        c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        return True if c.fetchone() else False

    def update_user(self, u: User):
        # TODO maybe update timestamp here?
        c = self.conn.cursor()
        c.execute("UPDATE users SET\
                   qrep=?, \
                   skill=?, \
                   repetition=?, \
                   last_study_time=?, \
                   scheduled_time=?, \
                   sm2_efactor=?, \
                   sm2_interval=?, \
                   leitner_box=?, \
                   last_update=?\
                   WHERE user_id=?", (
            json.dumps(u.qrep.tolist()),
            json.dumps(u.skill),
            json.dumps(u.repetition),
            json.dumps({k: str(v) for k, v in u.last_study_time.items()}),
            json.dumps({k: str(v) for k, v in u.scheduled_time.items()}),
            json.dumps(u.sm2_efactor),
            json.dumps(u.sm2_interval),
            json.dumps(u.leitner_box),
            u.last_update,
            u.user_id))
        self.conn.commit()

    def check_card(self, card_id: str) -> bool:
        c = self.conn.cursor()
        c.execute("SELECT * FROM cards WHERE card_id=?", (card_id,))
        return True if c.fetchone() else False

    def add_card(self, c: Card):
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO cards VALUES (?,?,?,?,?,?,?)',
                        (
                            c.card_id,
                            c.text,
                            c.answer,
                            json.dumps(c.qrep.tolist()),
                            str(c.skill),
                            c.category,
                            c.last_update))
        except sqlite3.IntegrityError:
            logger.info("card {} exists".format(c.card_id))
            print("card {} exists".format(c.card_id))
        self.conn.commit()

    def get_card(self, card_id: str = None):
        def row_to_dict(r):
            return Card(
                card_id=r[0],
                text=r[1],
                answer=r[2],
                qrep=np.array(json.loads(r[3])),
                skill=float(r[4]),
                category=r[5],
                last_update=r[6])
        c = self.conn.cursor()
        if card_id is None:
            c.execute("SELECT * FROM cards")
            return [row_to_dict(r) for r in c.fetchall()]
        else:
            c.execute("SELECT * FROM cards WHERE card_id=?", (card_id,))
            r = c.fetchone()
            return row_to_dict(r) if r else None

    def update_card(self, c: Card):
        # TODO maybe update timestamp here?
        cur = self.conn.cursor()
        cur.execute("UPDATE cards SET \
                     text=?, \
                     answer=?, \
                     qrep=?, \
                     skill=?, \
                     category=?, \
                     last_update=? \
                     WHERE card_id=?", (
            c.text,
            c.answer,
            json.dumps(c.qrep.tolist()),
            str(c.skill),
            c.category,
            c.last_update,
            c.card_id))
        self.conn.commit()

    def add_history(self, h: History):
        # TODO maybe choose timestamp here?
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (
                            h.history_id,
                            h.user_id,
                            h.card_id,
                            h.response,
                            h.judgement,
                            h.user_snapshot,
                            h.scheduler_snapshot,
                            json.dumps(h.card_ids),
                            h.scheduler_output,
                            h.timestamp))
        except sqlite3.IntegrityError:
            logger.info("history {} exists".format(h.history_id))
            print("history {} exists".format(h.history_id))
        self.conn.commit()

    def get_history(self, history_id: str = None):
        def row_to_dict(r):
            return History(
                history_id=r[0],
                user_id=r[1],
                card_id=r[2],
                response=r[3],
                judgement=r[4],
                user_snapshot=r[5],
                scheduler_snapshot=r[6],
                card_ids=json.loads(r[7]),
                scheduler_output=r[8],
                timestamp=r[9])
        c = self.conn.cursor()
        if history_id is None:
            c.execute("SELECT * FROM history")
            return [row_to_dict(r) for r in c.fetchall()]
        else:
            c.execute("SELECT * FROM history WHERE history_id=?", (history_id,))
            r = c.fetchone()
            return row_to_dict(r) if r else None

def test_user(db):
    user = User(
        user_id='user 1',
        qrep=np.array([0.1, 0.2, 0.3]),
        skill=[0.1, 0.2, 0.3],
        repetition={'card 1': 10},
        last_study_time={'card 1': datetime.now()},
        scheduled_time={'card 2': datetime.now()},
        sm2_efactor={'card 1': 0.5},
        sm2_interval={'card 1': 6},
        leitner_box={'card 1': 2},
        last_update=datetime.now()
    )
    print(user.to_snapshot())
    db.add_user(user)
    print()
    print(db.get_user())
    user = User(
        user_id='user 1',
        qrep=np.array([0.7, 0.8, 0.9]),
        skill=[0.7, 0.8, 0.9],
        repetition={'card 1': 11, 'card 2': 1},
        last_study_time={'card 1': datetime.now()},
        scheduled_time={'card 2': datetime.now()},
        sm2_efactor={'card 1': 0.5},
        sm2_interval={'card 1': 6},
        leitner_box={'card 1': 2},
        last_update=datetime.now()
    )
    db.update_user(user)
    print()
    print(db.get_user('user 1'))

def test_card(db):
    card = Card(
        card_id='card 1',
        text='This is the question text',
        answer='Answer Text III',
        qrep=np.array([1, 2, 3, 4]),
        skill=0.7,
        category='WORLD',
        last_update=datetime.now()
    )
    db.add_card(card)
    print(db.get_card('card 1'))
    card = Card(
        card_id='card 1',
        text='This is the NEWWWWWWW question text',
        answer='Answer Text IVVVV',
        qrep=np.array([1, 2, 3, 4]),
        skill=0.7,
        category='WORLD',
        last_update=datetime.now()
    )
    db.update_card(card)
    print()
    print(db.get_card('card 1'))

def test_history(db):
    user = User(
        user_id='user 1',
        qrep=np.array([0.1, 0.2, 0.3]),
        skill=[0.1, 0.2, 0.3],
        repetition={'card 1': 10},
        last_study_time={'card 1': datetime.now()},
        scheduled_time={'card 2': datetime.now()},
        sm2_efactor={'card 1': 0.5},
        sm2_interval={'card 1': 6},
        leitner_box={'card 1': 2},
        last_update=datetime.now()
    )
    card = Card(
        card_id='card 1',
        text='This is the question text',
        answer='Answer Text III',
        qrep=np.array([1, 2, 3, 4]),
        skill=0.7,
        category='WORLD',
        last_update=datetime.now()
    )
    params = Params()
    history = History(
        history_id='history 1',
        user_id=user.user_id,
        card_id=card.card_id,
        response='User Guess',
        judgement='wrong',
        user_snapshot=user.to_snapshot(),
        scheduler_snapshot=json.dumps(params.__dict__),
        card_ids=json.dumps([1, 2, 3, 4, 5]),
        scheduler_output='(awd, awd, awd)',
        timestamp=datetime.now())
    db.add_history(history)
    print()
    h = db.get_history()[0]
    print(h)
    u = User.from_snapshot(h.user_snapshot)
    print(u)

def test(db):
    test_user(db)
    print()
    print()
    print()
    test_card(db)
    print()
    print()
    print()
    test_history(db)


if __name__ == '__main__':
    db = SchedulerDB()
    print(db.get_card())
