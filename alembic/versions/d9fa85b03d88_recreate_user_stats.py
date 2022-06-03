"""recreate user stats

Revision ID: d9fa85b03d88
Revises: b179baed0aef
Create Date: 2022-06-03 01:29:16.915274

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Table, MetaData
from sqlalchemy import Column, ForeignKey, Integer, Float, Boolean, String, TIMESTAMP, Date

import pytz
import json
from tqdm import tqdm
from karl.db.session import SessionLocal, engine
from karl.models import StudyRecord, User, UserStatsV2
from karl.scheduler import KARLScheduler


# revision identifiers, used by Alembic.
revision = 'd9fa85b03d88'
down_revision = 'b179baed0aef'
branch_labels = None
depends_on = None


def update_user_stats(record, deck_id, utc_date, user_stats_table):
    if record.user_id not in user_stats_table:
        user_stats_table[record.user_id] = dict()
        curr_stats = None
    elif deck_id not in user_stats_table[record.user_id]:
        user_stats_table[record.user_id][deck_id] = dict()
        curr_stats = None
    else:
        curr_stats = user_stats_table[record.user_id][deck_id]

    is_new_stat = False
    if curr_stats is None:
        stats_id = json.dumps({
            'user_id': record.user_id,
            'deck_id': deck_id,
            'date': str(utc_date),
        })
        curr_stats = UserStatsV2(
            id=stats_id,
            user_id=record.user_id,
            deck_id=deck_id,
            date=utc_date,
            n_cards_total=0,
            n_cards_positive=0,
            n_new_cards_total=0,
            n_old_cards_total=0,
            n_new_cards_positive=0,
            n_old_cards_positive=0,
            elapsed_milliseconds_text=0,
            elapsed_milliseconds_answer=0,
            n_days_studied=0,
        )
        is_new_stat = True
    elif utc_date != curr_stats.date:
        # there is a previous user_stat, but not from today
        # copy user stat to today
        stats_id = json.dumps({
            'user_id': record.user_id,
            'deck_id': deck_id,
            'date': str(utc_date),
        })
        new_stat = UserStatsV2(
            id=stats_id,
            user_id=record.user_id,
            deck_id=deck_id,
            date=utc_date,
            n_cards_total=curr_stats.n_cards_total,
            n_cards_positive=curr_stats.n_cards_positive,
            n_new_cards_total=curr_stats.n_new_cards_total,
            n_old_cards_total=curr_stats.n_old_cards_total,
            n_new_cards_positive=curr_stats.n_new_cards_positive,
            n_old_cards_positive=curr_stats.n_old_cards_positive,
            elapsed_milliseconds_text=curr_stats.elapsed_milliseconds_text,
            elapsed_milliseconds_answer=curr_stats.elapsed_milliseconds_answer,
            n_days_studied=curr_stats.n_days_studied + 1,
        )
        curr_stats = new_stat
        is_new_stat = True

    if record.count == 0:
        curr_stats.n_new_cards_total += 1
        curr_stats.n_new_cards_positive += record.label
    else:
        curr_stats.n_old_cards_total += 1
        curr_stats.n_old_cards_positive += record.label

    curr_stats.n_cards_total += 1
    curr_stats.n_cards_positive += record.label
    curr_stats.elapsed_milliseconds_text += record.elapsed_milliseconds_text
    curr_stats.elapsed_milliseconds_answer += record.elapsed_milliseconds_answer

    user_stats_table[record.user_id][deck_id] = curr_stats

    return curr_stats


def upgrade():
    UserStatsV2.__table__.drop(engine)
    meta = MetaData()

    Table(
        'userstatsv2', meta,
        Column('id', String, primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id), index=True),
        Column('deck_id', String, nullable=False, index=True),
        Column('date', Date, nullable=False),
        Column('n_cards_total', Integer, nullable=False, default=0),
        Column('n_cards_positive', Integer, nullable=False, default=0),
        Column('n_new_cards_total', Integer, nullable=False, default=0),
        Column('n_old_cards_total', Integer, nullable=False, default=0),
        Column('n_new_cards_positive', Integer, nullable=False, default=0),
        Column('n_old_cards_positive', Integer, nullable=False, default=0),
        Column('elapsed_milliseconds_text', Integer, nullable=False, default=0),
        Column('elapsed_milliseconds_answer', Integer, nullable=False, default=0),
        Column('n_days_studied', Integer, nullable=False, default=0),
    )

    meta.create_all(engine)

    session = SessionLocal()

    user_stats_table = dict()
    user_stats_to_insert = dict()
    records = session.query(StudyRecord).order_by(StudyRecord.date)
    for record in tqdm(records, total=records.count()):
        utc_date = record.date.astimezone(pytz.utc).date()
        user_stats = update_user_stats(record, 'all', utc_date, user_stats_table)
        user_stats_to_insert[user_stats.id] = user_stats
            
        if record.deck_id is not None:
            user_stats = update_user_stats(record, record.deck_id, utc_date, user_stats_table)
            user_stats_to_insert[user_stats.id] = user_stats

    session.bulk_save_objects(user_stats_to_insert.values())
    session.commit()
    session.close()


def downgrade():
    pass
