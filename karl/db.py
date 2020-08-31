#!/usr/bin/env python
# coding: utf-8

import os
import json
import sqlite3
import logging
import pytz
from datetime import datetime
from typing import List
from dateutil.parser import parse as parse_date

from karl.util import User, Fact, History

logger = logging.getLogger('scheduler')

def copy_database(source_connection, dest_dbname=':memory:'):
    '''Return a connection to a new copy of an existing database.
       Raises an sqlite3.OperationalError if the destination already exists.
    '''
    script = ''.join(source_connection.iterdump())
    dest_conn = sqlite3.connect(dest_dbname,
                                detect_types=sqlite3.PARSE_DECLTYPES,
                                check_same_thread=False)
    dest_conn.executescript(script)
    return dest_conn

class SchedulerDB:

    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(filename):
            self.create()

        # load DB from disk
        source = sqlite3.connect(
            filename,
            detect_types=sqlite3.PARSE_DECLTYPES,
            check_same_thread=False
        )
        self.conn = copy_database(source, ':memory:')

        # TODO see if this helps
        cur = self.conn.cursor()
        cur.execute('CREATE INDEX index_user_id ON history (user_id);')

        self.conn.commit()
        source.close()

    def create(self):
        logger.info('creating {}'.format(self.filename))
        conn = sqlite3.connect(self.filename)
        cur = conn.cursor()

        # *current* state of the player
        # including leitner and sm2 info
        cur.execute('CREATE TABLE users (\
                     user_id PRIMARY KEY, \
                     recent_facts TEXT, \
                     previous_study TEXT, \
                     leitner_box TEXT, \
                     leitner_scheduled_date TEXT, \
                     sm2_efactor TEXT, \
                     sm2_interval TEXT, \
                     sm2_repetition TEXT, \
                     sm2_scheduled_date TEXT, \
                     results TEXT, \
                     count_correct_before TEXT, \
                     count_wrong_before TEXT, \
                     params TEXT)'
                    )

        # *current* cache of facts
        cur.execute('CREATE TABLE facts(\
                     fact_id PRIMARY KEY, \
                     text TEXT, \
                     answer TEXT, \
                     category TEXT, \
                     deck_name TEXT, \
                     deck_id TEXT, \
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
                     debug_id TEXT, \
                     user_id TEXT, \
                     fact_id TEXT, \
                     deck_id TEXT, \
                     response TEXT, \
                     judgement TEXT, \
                     user_snapshot TEXT, \
                     scheduler_snapshot TEXT, \
                     fact_ids TEXT, \
                     scheduler_output TEXT, \
                     elapsed_seconds_text INT, \
                     elapsed_seconds_answer INT, \
                     elapsed_milliseconds_text INT, \
                     elapsed_milliseconds_answer INT, \
                     is_new_fact INT, \
                     date timestamp)'
                    )

        conn.commit()
        conn.close()

    def add_user(self, u: User):
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO users VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', u.pack())
        except sqlite3.IntegrityError:
            logger.info("user {} exists".format(u.user_id))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def get_user(self, user_id=None):
        cur = self.conn.cursor()
        if user_id is None:
            cur.execute("SELECT * FROM users")
            return [User.unpack(r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
            r = cur.fetchone()
            return User.unpack(r) if r else None

    def check_user(self, user_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
        return True if cur.fetchone() else False

    def update_user(self, u: User):
        cur = self.conn.cursor()
        u = u.pack()
        u = u[1:] + u[:1]  # move user_id to the end
        cur.execute("UPDATE users SET\
                     recent_facts=?, \
                     previous_study=?, \
                     leitner_box=?, \
                     leitner_scheduled_date=?, \
                     sm2_efactor=?, \
                     sm2_interval=?, \
                     sm2_repetition=?, \
                     sm2_scheduled_date=?, \
                     results=?, \
                     count_correct_before=?, \
                     count_wrong_before=?, \
                     params=? \
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
            cur.execute('INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)', c.pack())
        except sqlite3.IntegrityError:
            logger.info("fact {} exists".format(c.fact_id))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def add_facts(self, facts: List[Fact]):
        cur = self.conn.cursor()
        for c in facts:
            try:
                cur.execute('INSERT INTO facts VALUES (?,?,?,?,?,?,?,?,?)', c.pack())
            except sqlite3.IntegrityError:
                logger.info("fact {} exists".format(c.fact_id))
        # NOTE web.py will commit at exit
        # self.conn.commit()

    def get_fact(self, fact_id=None):
        cur = self.conn.cursor()
        if fact_id is None:
            cur.execute("SELECT * FROM facts")
            return [Fact.unpack(r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT * FROM facts WHERE fact_id=?", (fact_id,))
            r = cur.fetchone()
            return Fact.unpack(r) if r else None

    def update_fact(self, c: Fact):
        cur = self.conn.cursor()
        c = c.pack()
        c = c[1:] + c[:1]  # move fact_id to the end
        cur.execute("UPDATE facts SET \
                     text=?, \
                     answer=?, \
                     category=?, \
                     deck_name=?, \
                     deck_id=?, \
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
        # convert all date to UTC
        date = h.date.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
        cur = self.conn.cursor()
        try:
            cur.execute('INSERT INTO history VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                        (
                            h.history_id,
                            h.debug_id,
                            h.user_id,
                            h.fact_id,
                            h.deck_id,
                            h.response,
                            h.judgement,
                            h.user_snapshot,
                            h.scheduler_snapshot,
                            json.dumps(h.fact_ids),
                            h.scheduler_output,
                            h.elapsed_seconds_text,
                            h.elapsed_seconds_answer,
                            h.elapsed_milliseconds_text,
                            h.elapsed_milliseconds_answer,
                            h.is_new_fact,
                            date,
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
        cur = self.conn.cursor()
        if history_id is None:
            cur.execute("SELECT * FROM history")
            return [History(*r) for r in cur.fetchall()]
        else:
            cur.execute("SELECT * FROM history WHERE history_id=?", (history_id,))
            r = cur.fetchone()
            return History(*r) if r else None

    def get_user_history(self, user_id: str, deck_id: str = None,
                         date_start: str = None, date_end: str = None):
        cur = self.conn.cursor()
        if date_start is None:
            date_start = '2008-06-11T08:00:00Z'
        if date_end is None:
            date_end = '2038-06-11T08:00:00Z'
        # convert all date to UTC
        date_start = parse_date(date_start).astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
        date_end = parse_date(date_end).astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')

        if deck_id is not None:
            cur.execute("SELECT * FROM history WHERE user_id=? AND deck_id=? AND date BETWEEN ? AND ?",
                        (user_id, deck_id, date_start, date_end,))
        else:
            cur.execute("SELECT * FROM history WHERE user_id=? AND date BETWEEN ? AND ?",
                        (user_id, date_start, date_end,))
        return [History(*r) for r in cur.fetchall()]

    def check_history(self, history_id: str) -> bool:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM history WHERE history_id=?", (history_id,))
        return True if cur.fetchone() else False

    def update_history(self, old_history_id: str, h: History):
        '''This is different, `history_id` might get updated'''
        # convert all date to UTC
        date = h.date.astimezone(pytz.utc).strftime('%Y-%m-%d %H:%M:%S')
        cur = self.conn.cursor()
        if h.history_id == old_history_id:
            # inplace update with same id
            cur.execute("UPDATE history SET \
                         debug_id=?, \
                         user_id=?, \
                         fact_id=?, \
                         deck_id=?, \
                         response=?, \
                         judgement=?, \
                         user_snapshot=?, \
                         scheduler_snapshot=?, \
                         fact_ids=?, \
                         scheduler_output=?, \
                         elapsed_seconds_text=?, \
                         elapsed_seconds_answer=?, \
                         elapsed_milliseconds_text=?, \
                         elapsed_milliseconds_answer=?, \
                         is_new_fact=?, \
                         date=? \
                         WHERE history_id=?", (
                h.debug_id,
                h.user_id,
                h.fact_id,
                h.deck_id,
                h.response,
                h.judgement,
                h.user_snapshot,
                h.scheduler_snapshot,
                json.dumps(h.fact_ids),
                h.scheduler_output,
                h.elapsed_seconds_text,
                h.elapsed_seconds_answer,
                h.elapsed_milliseconds_text,
                h.elapsed_milliseconds_answer,
                h.is_new_fact,
                date,
                h.history_id))
        else:
            # replace update with new id
            cur.execute("UPDATE history SET \
                         debug_id=?, \
                         history_id=?, \
                         user_id=?, \
                         fact_id=?, \
                         deck_id=?, \
                         response=?, \
                         judgement=?, \
                         user_snapshot=?, \
                         scheduler_snapshot=?, \
                         fact_ids=?, \
                         scheduler_output=?, \
                         elapsed_seconds_text=?, \
                         elapsed_seconds_answer=?, \
                         elapsed_milliseconds_text=?, \
                         elapsed_milliseconds_answer=?, \
                         is_new_fact=?, \
                         date=? \
                         WHERE history_id=?", (
                h.debug_id,
                h.history_id,
                h.user_id,
                h.fact_id,
                h.deck_id,
                h.response,
                h.judgement,
                h.user_snapshot,
                h.scheduler_snapshot,
                json.dumps(h.fact_ids),
                h.scheduler_output,
                h.elapsed_seconds_text,
                h.elapsed_seconds_answer,
                h.elapsed_milliseconds_text,
                h.elapsed_milliseconds_answer,
                h.is_new_fact,
                date,
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
        # new_filename = '{}_{}'.format(self.filename, datetime.now())
        # copy_database(self.conn, new_filename).close()
        self.conn.close()
        logger.info('db commit')
