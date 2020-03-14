import os
import json
import sqlite3
import logging
import numpy as np
from datetime import datetime
from util import User, Card, History

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
                prob TEXT, \
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
                          json.dumps(u.skill.tolist()),
                          json.dumps(u.repetition),
                          json.dumps({k: str(v) for k, v in u.last_study_time.items()}),
                          json.dumps({k: str(v) for k, v in u.scheduled_time.items()}),
                          json.dumps(u.sm2_efactor),
                          json.dumps(u.sm2_interval),
                          json.dumps(u.leitner_box),
                          u.last_update
                      ))
        except sqlite3.IntegrityError:
            logger.info("user {} exists".format(u.user_id))
        self.conn.commit()

    def get_user(self, user_id=None):
        def row_to_dict(r):
            return User(
                user_id=r[0],
                qrep=np.array(json.loads(r[1])),
                skill=np.array(json.loads(r[2])),
                repetition=json.loads(r[3]),
                last_study_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                                 for k, v in json.loads(r[4]).items()},
                scheduled_time={k: datetime.strptime(v, "%Y-%m-%d %H:%M:%S.%f")
                                for k, v in json.loads(r[5]).items()},
                sm2_efactor=json.loads(r[6]),
                sm2_interval=json.loads(r[7]),
                leitner_box=json.loads(r[8]),
                last_update=r[9]
            )
        c = self.conn.cursor()
        if user_id is None:
            c.execute("SELECT * FROM users")
            return [row_to_dict(r) for r in c.fetchall()]
        else:
            c.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
            r = c.fetchall()
            if len(r) == 0:
                return None
            else:
                return row_to_dict(r[0])
    '''
    def update_player(self, player):
        c = self.conn.cursor()
        c.execute("UPDATE players SET score=?,\
                questions_seen=?,questions_answered=?,questions_correct=? \
                WHERE player_id=?", (
                    player.score,
                    json.dumps(player.questions_seen),
                    json.dumps(player.questions_answered),
                    json.dumps(player.questions_correct),
                    player.uid))
        self.conn.commit()
    '''


if __name__ == '__main__':
    db = SchedulerDB()
    user = User(
        user_id='user 1',
        qrep=np.array([1, 2, 3]),
        skill=np.array([0.1, 0.2, 0.3]),
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
