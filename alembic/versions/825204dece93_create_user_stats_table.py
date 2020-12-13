"""create user stats table

Revision ID: 825204dece93
Revises: 285f19cbc355
Create Date: 2020-12-13 02:51:29.523284

"""
import pytz
import json
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, String, Integer, Date

from karl.models import User, UserStats


# revision identifiers, used by Alembic.
revision = '825204dece93'
down_revision = '285f19cbc355'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'userstats',
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


def schema_downgrade():
    op.drop_table('userstats')


def update_user_stats(session, user, record, deck_id):
    # get the latest user_stat ordered by date
    curr_stat = session.query(UserStats).\
        filter(UserStats.user_id == user.id).\
        filter(UserStats.deck_id == deck_id).\
        order_by(UserStats.date.desc()).first()

    # NOTE we consider UTC days
    date = record.date.astimezone(pytz.utc).date()

    is_new_stat = False
    if curr_stat is None:
        stats_id = json.dumps({
            'user_id': user.id,
            'deck_id': deck_id,
            'date': str(date),
        })
        curr_stat = UserStats(
            id=stats_id,
            user_id=user.id,
            deck_id=deck_id,
            date=date,
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

    if date != curr_stat.date:
        # there is a previous user_stat, but not from today
        # copy user stat to today
        stats_id = json.dumps({
            'user_id': user.id,
            'deck_id': deck_id,
            'date': str(date),
        })
        new_stat = UserStats(
            id=stats_id,
            user_id=user.id,
            deck_id=deck_id,
            date=date,
            n_cards_total=curr_stat.n_cards_total,
            n_cards_positive=curr_stat.n_cards_positive,
            n_new_cards_total=curr_stat.n_new_cards_total,
            n_old_cards_total=curr_stat.n_old_cards_total,
            n_new_cards_positive=curr_stat.n_new_cards_positive,
            n_old_cards_positive=curr_stat.n_old_cards_positive,
            elapsed_milliseconds_text=curr_stat.elapsed_milliseconds_text,
            elapsed_milliseconds_answer=curr_stat.elapsed_milliseconds_answer,
            n_days_studied=curr_stat.n_days_studied + 1,
        )
        curr_stat = new_stat
        is_new_stat = True

    if record.is_new_fact:
        curr_stat.n_new_cards_total += 1
        curr_stat.n_new_cards_positive += record.response
    else:
        curr_stat.n_old_cards_total += 1
        curr_stat.n_old_cards_positive += record.response

    curr_stat.n_cards_total += 1
    curr_stat.n_cards_positive += record.response
    curr_stat.elapsed_milliseconds_text += record.elapsed_milliseconds_text
    curr_stat.elapsed_milliseconds_answer += record.elapsed_milliseconds_answer

    if is_new_stat:
        session.add(curr_stat)


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session.query(User)
    for user in tqdm(users, total=users.count()):
        # go through all records of each user
        # save the user's statistics at the end of each UTC day
        for record in user.records:
            if record.deck_id is not None:
                update_user_stats(session, user, record, record.deck_id)
            update_user_stats(session, user, record, 'all')
    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
