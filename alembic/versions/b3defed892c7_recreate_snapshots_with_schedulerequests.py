"""recreate snapshots with schedulerequests

Revision ID: b3defed892c7
Revises: dd21f1d7dbcf
Create Date: 2022-06-01 22:27:56.581888

"""
from tqdm import tqdm
from sqlalchemy import Column, ForeignKey, Integer, Float, Boolean, String, TIMESTAMP
from sqlalchemy import Table, MetaData
from sqlalchemy.dialects.postgresql import JSONB

from karl.models import UserFeatureVector, CardFeatureVector, UserCardFeatureVector
from karl.models import UserSnapshot, CardSnapshot, UserCardSnapshot
from karl.models import User, Card, Record, ScheduleRequest
from karl.db.session import SessionLocal, engine


# revision identifiers, used by Alembic.
revision = 'b3defed892c7'
down_revision = 'dd21f1d7dbcf'
branch_labels = None
depends_on = None


def upgrade():
    meta = MetaData()

    Table(
        'usercardsnapshot', meta,
        Column('id', String, ForeignKey(ScheduleRequest.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id, ondelete='CASCADE'), index=True),
        Column('card_id', String, ForeignKey(Card.id, ondelete='CASCADE'), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('n_study_positive_session', Integer),
        Column('n_study_negative_session', Integer),
        Column('n_study_total_session', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('delta_session', Integer),
        Column('previous_delta_session', Integer),
        Column('previous_study_date_session', TIMESTAMP(timezone=True)),
        Column('previous_study_response_session', Boolean),
        Column('leitner_box', Integer),
        Column('leitner_scheduled_date', TIMESTAMP(timezone=True)),
        Column('sm2_efactor', Float),
        Column('sm2_interval', Float),
        Column('sm2_repetition', Integer),
        Column('sm2_scheduled_date', TIMESTAMP(timezone=True)),
        Column('correct_on_first_try', Boolean),
        Column('correct_on_first_try_session', Boolean),
    )

    Table(
        'usersnapshot', meta,
        Column('id', String, ForeignKey(Record.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id, ondelete='CASCADE'), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('n_study_positive_session', Integer),
        Column('n_study_negative_session', Integer),
        Column('n_study_total_session', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('delta_session', Integer),
        Column('previous_delta_session', Integer),
        Column('previous_study_date_session', TIMESTAMP(timezone=True)),
        Column('previous_study_response_session', Boolean),
        Column('parameters', JSONB),
    )

    Table(
        'cardsnapshot', meta,
        Column('id', String, ForeignKey(Record.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id, ondelete='CASCADE'), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    meta.create_all(engine)

    session = SessionLocal()

    '''
    usercardfeaturevectors = session.query(UserCardFeatureVector).order_by(UserCardFeatureVector.date)
    usercardsnapshots = []
    schedule_request_ids = set()
    for v in tqdm(usercardfeaturevectors, total=usercardfeaturevectors.count()):
        r = session.query(Record).get(v.id)
        if r is None:
            continue
        if v.id in schedule_request_ids:
            continue
        schedule_request_ids.add(v.id)

        d = v.__dict__
        d.update({
            'n_study_positive_session': 0,
            'n_study_negative_session': 0,
            'n_study_total_session': 0,
            'delta_session': None,
            'previous_delta_session': None,
            'previous_study_date_session': None,
            'previous_study_response_session': None,
            'correct_on_first_try_session': False,
        })
        usercardsnapshots.append(d)
    session.bulk_insert_mappings(UserCardSnapshot, usercardsnapshots)
    session.commit()

    userfeaturevectors = session.query(UserFeatureVector).order_by(UserFeatureVector.date)
    usersnapshots = []
    schedule_request_ids = set()
    for v in tqdm(userfeaturevectors, total=userfeaturevectors.count()):
        r = session.query(Record).get(v.id)
        if r is None:
            continue
        if v.id in schedule_request_ids:
            continue
        schedule_request_ids.add(v.id)

        d = v.__dict__
        d.update({
            'n_study_positive_session': 0,
            'n_study_negative_session': 0,
            'n_study_total_session': 0,
            'delta_session': None,
            'previous_delta_session': None,
            'previous_study_date_session': None,
            'previous_study_response_session': None,
        })
        usersnapshots.append(d)
    session.bulk_insert_mappings(UserSnapshot, usersnapshots)
    session.commit()
    '''

    cardfeaturevectors = session.query(CardFeatureVector).order_by(CardFeatureVector.date)
    cardsnapshots = []
    schedule_request_ids = set()
    for v in tqdm(cardfeaturevectors, total=cardfeaturevectors.count()):
        r = session.query(Record).get(v.id)
        if r is None:
            continue
        if v.id in schedule_request_ids:
            continue
        schedule_request_ids.add(v.id)

        d = v.__dict__
        cardsnapshots.append(d)
    session.bulk_insert_mappings(CardSnapshot, cardsnapshots)
    session.commit()

def downgrade():
    pass
