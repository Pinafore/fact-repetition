#!/usr/bin/env python
# coding: utf-8

import os
import json
import sqlite3
import logging
from typing import List

from karl.util import User, Fact, History

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
                     previous_study TEXT, \
                     leitner_box TEXT, \
                     leitner_scheduled_date TEXT, \
                     sm2_efactor TEXT, \
                     sm2_interval TEXT, \
                     sm2_repetition TEXT, \
                     sm2_scheduled_date TEXT, \
                     results TEXT, \
                     count_correct_before TEXT, \
                     count_wrong_before TEXT)'
                    )

        # *current* cache of facts 
        cur.execute('CREATE TABLE facts(\
                     fact_id PRIMARY KEY, \
                     text TEXT, \
                     answer TEXT, \
                     category TEXT, \
                     qrep TEXT, \
                     skill TEXT, \
                     results TEXT)'
                    )

        # input to and output from the schedule API
        # at the end of each schedule API call we write to the history table
        # with a placeholder history_id, then during update API call we update
        # it with the history_id from the front end DB.
        cur.execute('CREATE TABLE history (\
                     history_id PRIMARY KEY, \
                     user_id TEXT, \
                     fact_id TEXT, \
                     response TEXT, \
                     judgement TEXT, \
                     user_snapshot TEXT, \
                     scheduler_snapshot TEXT, \
                     fact_ids TEXT, \
                     scheduler_output TEXT, \
                     date timestamp)'
                    )

        conn.commit()
        conn.close()

    def add_user(self, u: User):
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)', u.pack())
        except sqlite3.IntegrityError:
            logger.info("user {} exists".format(u.user_id))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def get_user(self, user_id=None):
        def row_to_dict(r):
            return User.unpack(r)

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
        u = u.pack()
        u = u[1:] + u[:1]  # move user_id to the end
        cur.execute("UPDATE users SET\
                     qrep=?, \
                     skill=?, \
                     category=?, \
                     previous_study=?, \
                     leitner_box=?, \
                     leitner_scheduled_date=?, \
                     sm2_efactor=?, \
                     sm2_interval=?, \
                     sm2_repetition=?, \
                     sm2_scheduled_date=?, \
                     results=?, \
                     count_correct_before=?, \
                     count_wrong_before=? \
                     WHERE user_id=?",
                    u)
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def delete_user(self, user_id=None):
        cur = self.conn.cursor()
        if user_id is None:
            logger.info('deleting all users from db')
            cur.execute("DELETE FROM users")
        else:
            logger.info('deleting user {} from db'.format(user_id))
            cur.execute("DELETE FROM users WHERE user_id=?", (user_id,))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def check_fact(self, fact_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,))
        return True if cur.fetchone() else False

    def add_fact(self, c: Fact):
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO facts VALUES (?,?,?,?,?,?,?)', c.pack())
        except sqlite3.IntegrityError:
            logger.info("fact {} exists".format(c.fact_id))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def add_facts(self, facts: List[Fact]):
        cur = self.conn.cursor()
        for c in facts:
            try:
                cur.execute('INSERT INTO facts VALUES (?,?,?,?,?,?,?)', c.pack())
            except sqlite3.IntegrityError:
                logger.info("fact {} exists".format(c.fact_id))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def get_fact(self, fact_id=None):
        def row_to_dict(r):
            return Fact.unpack(r)

        cur = self.conn.cursor()
        if fact_id is None:
            cur.execute("SELECT * FROM facts")
            return [row_to_dict(r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,))
            r = cur.fetchone()
            return row_to_dict(r) if r else None

    def update_fact(self, c: Fact):
        cur = self.conn.cursor()
        c = c.pack()
        c = c[1:] + c[:1]  # move fact_id to the end
        cur.execute("UPDATE facts SET \
                     text=?, \
                     answer=?, \
                     category=?, \
                     qrep=?, \
                     skill=?, \
                     results=? \
                     WHERE fact_id=?",
                    c)
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def delete_fact(self, fact_id: str):
        cur = self.conn.cursor()
        if fact_id is None:
            logger.info('deleting all facts from db')
            cur.execute("DELETE FROM facts")
        else:
            logger.info('deleting fact {} from db'.format(fact_id))
            cur.execute("DELETE FROM facts WHERE fact_id=?", (fact_id,))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def add_history(self, h: History):
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?)',
                        (
                            h.history_id,
                            h.user_id,
                            h.fact_id,
                            h.response,
                            h.judgement,
                            h.user_snapshot,
                            h.scheduler_snapshot,
                            json.dumps(h.fact_ids),
                            h.scheduler_output,
                            h.date
                        ))
        except sqlite3.IntegrityError:
            # this means the fact was shown to user but we didn't receive a
            # response, replace
            logger.info("history {} exists, replacing".format(h.history_id))
            self.delete_history(h.history_id)
            self.add_history(h)
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def get_history(self, history_id=None):
        def row_to_dict(r):
            return History(
                history_id=r[0],
                user_id=r[1],
                fact_id=r[2],
                response=r[3],
                judgement=r[4],
                user_snapshot=r[5],
                scheduler_snapshot=r[6],
                fact_ids=json.loads(r[7]),
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
        if h.history_id == old_history_id:
            # inplace update with same id
            cur.execute("UPDATE history SET \
                         user_id=?, \
                         fact_id=?, \
                         response=?, \
                         judgement=?, \
                         user_snapshot=?, \
                         scheduler_snapshot=?, \
                         fact_ids=?, \
                         scheduler_output=?, \
                         date=? \
                         WHERE history_id=?", (
                h.user_id,
                h.fact_id,
                h.response,
                h.judgement,
                h.user_snapshot,
                h.scheduler_snapshot,
                json.dumps(h.fact_ids),
                h.scheduler_output,
                h.date,
                h.history_id))
        else:
            # replace update with new id
            cur.execute("UPDATE history SET \
                         history_id=?, \
                         user_id=?, \
                         fact_id=?, \
                         response=?, \
                         judgement=?, \
                         user_snapshot=?, \
                         scheduler_snapshot=?, \
                         fact_ids=?, \
                         scheduler_output=?, \
                         date=? \
                         WHERE history_id=?", (
                h.history_id,
                h.user_id,
                h.fact_id,
                h.response,
                h.judgement,
                h.user_snapshot,
                h.scheduler_snapshot,
                json.dumps(h.fact_ids),
                h.scheduler_output,
                h.date,
                old_history_id))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def delete_history(self, history_id=None, user_id=None):
        cur = self.conn.cursor()
        if history_id is not None:
            logger.info('deleting history {} from db'.format(history_id))
            cur.execute("DELETE FROM history WHERE history_id=?", (history_id,))
        elif user_id is not None:
            logger.info('deleting history of user {} from db'.format(user_id))
            cur.execute("DELETE FROM history WHERE user_id=?", (user_id,))
        else:
            logger.info('deleting all history from db')
            cur.execute("DELETE FROM history")
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def finalize(self):
        self.conn.commit()
        self.conn.close()
        logger.info('db commit')
