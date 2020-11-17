"""correct new facts

Revision ID: 4db7644e1ece
Revises: e2b7e34d9144
Create Date: 2020-11-17 12:16:31.255246

"""
import json
from alembic import op
from sqlalchemy import orm

from karl.models import User, UserStat, Record


# revision identifiers, used by Alembic.
revision = '4db7644e1ece'
down_revision = 'e2b7e34d9144'
branch_labels = None
depends_on = None


def update_user_stats(
    session,
    user: User,
    record: Record,
    deck_id: str,
):
    # get the latest user_stat ordered by date
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
            n_days_studied=0,
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
            n_days_studied=curr_stat.n_days_studied + 1,
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

def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    session.query(UserStat).delete()

    for user in session.query(User):
        if len(user.records) == 0:
            continue

        is_new_fact = dict()  # fact_id -> True/False
        for record in user.records:
            if is_new_fact.get(record.fact_id, True):
                record.is_new_fact = True
            else:
                record.is_new_fact = False
            is_new_fact[record.fact_id] = False

            update_user_stats(session, user, record, deck_id='all')
            if record.deck_id is not None:
                update_user_stats(session, user, record, deck_id=record.deck_id)

    session.commit()


def downgrade():
    pass
