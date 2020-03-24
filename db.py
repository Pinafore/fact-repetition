import os
import json
import sqlite3
import logging
import numpy as np
from util import User, Card, History, parse_date

logger = logging.getLogger('scheduler')

class SchedulerDB:

    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(filename):
            self.create()
        self.conn = sqlite3.connect(
            filename, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)

    def create(self):
        logger.info('creating {}'.format(self.filename))
        conn = sqlite3.connect(self.filename)
        cur = conn.cursor()

        # *current* state of the player
        # including leitner and sm2 info
        cur.execute('CREATE TABLE users (\
                     user_id PRIMARY KEY, \
                     qrep TEXT, \
                     skill TEXT, \
                     category TEXT, \
                     last_study_time TEXT, \
                     leitner_box TEXT, \
                     leitner_scheduled_time TEXT, \
                     sm2_efactor TEXT, \
                     sm2_interval TEXT, \
                     sm2_repetition TEXT, \
                     sm2_scheduled_time TEXT, \
                     date timestamp)')

        # *current* cache of cards
        cur.execute('CREATE TABLE cards (\
                     card_id PRIMARY KEY, \
                     text TEXT, \
                     answer TEXT, \
                     qrep TEXT, \
                     skill TEXT, \
                     category TEXT, \
                     date timestamp)')

        # input to and output from the schedule API
        # at the end of each schedule API call we write to the history table
        # with a placeholder history_id, then during update API call we update
        # it with the history_id from the front end DB.
        cur.execute('CREATE TABLE history (\
                     history_id PRIMARY KEY, \
                     user_id TEXT, \
                     card_id TEXT, \
                     response TEXT, \
                     judgement TEXT, \
                     user_snapshot TEXT, \
                     scheduler_snapshot TEXT, \
                     card_ids TEXT, \
                     scheduler_output TEXT, \
                     date timestamp)')

        conn.commit()
        conn.close()

    def add_user(self, u: User):
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                        (
                            u.user_id,
                            json.dumps([x.tolist() for x in u.qrep]),
                            json.dumps([x.tolist() for x in u.skill]),
                            u.category,
                            json.dumps({k: str(v) for k, v in u.last_study_time.items()}),
                            json.dumps(u.leitner_box),
                            json.dumps({k: str(v) for k, v in u.leitner_scheduled_time.items()}),
                            json.dumps(u.sm2_efactor),
                            json.dumps(u.sm2_interval),
                            json.dumps(u.sm2_repetition),
                            json.dumps({k: str(v) for k, v in u.sm2_scheduled_time.items()}),
                            u.date))
        except sqlite3.IntegrityError:
            logger.info("user {} exists".format(u.user_id))
        self.conn.commit()

    def get_user(self, user_id: str = None):
        def row_to_dict(r):
            return User(
                user_id=r[0],
                qrep=[np.array(x) for x in json.loads(r[1])],
                skill=[np.array(x) for x in json.loads(r[2])],
                category=r[3],
                last_study_time={k: parse_date(v) for k, v in json.loads(r[4]).items()},
                leitner_box=json.loads(r[5]),
                leitner_scheduled_time={k: parse_date(v) for k, v in json.loads(r[6]).items()},
                sm2_efactor=json.loads(r[7]),
                sm2_interval=json.loads(r[8]),
                sm2_repetition=json.loads(r[9]),
                sm2_scheduled_time={k: parse_date(v) for k, v in json.loads(r[10]).items()},
                date=r[11])
        cur = self.conn.cursor()
        if user_id is None:
            cur.execute("SELECT * FROM users")
            return [row_to_dict(r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
            r = cur.fetchone()
            return row_to_dict(r) if r else None

    def check_user(self, user_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        return True if cur.fetchone() else False

    def update_user(self, u: User):
        cur = self.conn.cursor()
        cur.execute("UPDATE users SET\
                     qrep=?, \
                     skill=?, \
                     category=?, \
                     last_study_time=?, \
                     leitner_box=?, \
                     leitner_scheduled_time=?, \
                     sm2_efactor=?, \
                     sm2_interval=?, \
                     sm2_repetition=?, \
                     sm2_scheduled_time=?, \
                     date=? \
                     WHERE user_id=?", (
            json.dumps([x.tolist() for x in u.qrep]),
            json.dumps([x.tolist() for x in u.skill]),
            u.category,
            json.dumps({k: str(v) for k, v in u.last_study_time.items()}),
            json.dumps(u.leitner_box),
            json.dumps({k: str(v) for k, v in u.leitner_scheduled_time.items()}),
            json.dumps(u.sm2_efactor),
            json.dumps(u.sm2_interval),
            json.dumps(u.sm2_repetition),
            json.dumps({k: str(v) for k, v in u.sm2_scheduled_time.items()}),
            u.date,
            u.user_id))
        self.conn.commit()

    def delete_user(self, user_id: str = None):
        cur = self.conn.cursor()
        if user_id is None:
            logger.info('deleting all users from db')
            cur.execute("DELETE FROM users")
        else:
            logger.info('deleting user {} from db'.format(user_id))
            cur.execute("DELETE * FROM users WHERE user_id=?", (user_id,))
        self.conn.commit()

    def check_card(self, card_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM cards WHERE card_id=?", (card_id,))
        return True if cur.fetchone() else False

    def add_card(self, c: Card):
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO cards VALUES (?,?,?,?,?,?,?)',
                        (
                            c.card_id,
                            c.text,
                            c.answer,
                            json.dumps(c.qrep.tolist()),
                            json.dumps(c.skill.tolist()),
                            c.category,
                            c.date))
        except sqlite3.IntegrityError:
            logger.info("card {} exists".format(c.card_id))
        self.conn.commit()

    def get_card(self, card_id: str = None):
        def row_to_dict(r):
            return Card(
                card_id=r[0],
                text=r[1],
                answer=r[2],
                qrep=np.array(json.loads(r[3])),
                skill=np.array(json.loads(r[4])),
                category=r[5],
                date=r[6])
        cur = self.conn.cursor()
        if card_id is None:
            cur.execute("SELECT * FROM cards")
            return [row_to_dict(r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT * FROM cards WHERE card_id=?", (card_id,))
            r = cur.fetchone()
            return row_to_dict(r) if r else None

    def update_card(self, c: Card):
        cur = self.conn.cursor()
        cur.execute("UPDATE cards SET \
                     text=?, \
                     answer=?, \
                     qrep=?, \
                     skill=?, \
                     category=?, \
                     date=? \
                     WHERE card_id=?", (
            c.text,
            c.answer,
            json.dumps(c.qrep.tolist()),
            json.dumps(c.skill.tolist()),
            c.category,
            c.date,
            c.card_id))
        self.conn.commit()

    def delete_card(self, card_id: str):
        cur = self.conn.cursor()
        if card_id is None:
            logger.info('deleting all cards from db')
            cur.execute("DELETE FROM cards")
        else:
            logger.info('deleting card {} from db'.format(card_id))
            cur.execute("DELETE * FROM cards WHERE card_id=?", (card_id,))
        self.conn.commit()

    def add_history(self, h: History):
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
                            h.date))
        except sqlite3.IntegrityError:
            logger.info("history {} exists".format(h.history_id))
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
                date=r[9])
        cur = self.conn.cursor()
        if history_id is None:
            cur.execute("SELECT * FROM history")
            return [row_to_dict(r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT * FROM history WHERE history_id=?", (history_id,))
            r = cur.fetchone()
            return row_to_dict(r) if r else None

    def check_history(self, history_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM history WHERE history_id=?", (history_id,))
        return True if cur.fetchone() else False

    def update_history(self, old_history_id: str, h: History):
        '''This is different, `history_id` might get updated'''
        cur = self.conn.cursor()
        cur.execute("UPDATE history SET \
                     history_id=?, \
                     user_id=?, \
                     card_id=?, \
                     response=?, \
                     judgement=?, \
                     user_snapshot=?, \
                     scheduler_snapshot=?, \
                     card_ids=?, \
                     scheduler_output=?, \
                     date=? \
                     WHERE history_id=?", (
            h.history_id,
            h.user_id,
            h.card_id,
            h.response,
            h.judgement,
            h.user_snapshot,
            h.scheduler_snapshot,
            json.dumps(h.card_ids),
            h.scheduler_output,
            h.date,
            old_history_id))
        self.conn.commit()

    def delete_history(self, history_id: str = None):
        cur = self.conn.cursor()
        if history_id is None:
            logger.info('deleting all history from db')
            cur.execute("DELETE FROM history")
        else:
            logger.info('deleting history {} from db'.format(history_id))
            cur.execute("DELETE * FROM history WHERE history_id=?", (history_id,))
        self.conn.commit()
