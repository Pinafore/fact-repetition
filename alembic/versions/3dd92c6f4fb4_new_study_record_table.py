"""new study record table

Revision ID: 3dd92c6f4fb4
Revises: c63c0f83d6da
Create Date: 2022-06-01 20:40:25.978026

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Table, MetaData
from sqlalchemy import Column, ForeignKey, ARRAY, String, TIMESTAMP, Boolean, Integer
from tqdm import tqdm

from karl.models import Record, StudyRecord, ScheduleRequest
from karl.models import User, Card
from karl.db.session import SessionLocal, engine


# revision identifiers, used by Alembic.
revision = '3dd92c6f4fb4'
down_revision = 'c63c0f83d6da'
branch_labels = None
depends_on = None


def upgrade():
    StudyRecord.__table__.drop(engine)

    meta = MetaData()

    Table(
        'studyrecord', meta,
        Column('id', String, primary_key=True, index=True),  # history_id / front_end_id provided by Matthew
        Column('debug_id', String, ForeignKey(ScheduleRequest.id), index=True),
        Column('studyset_id', String, index=True),  # session id
        Column('user_id', String, ForeignKey(User.id), index=True),
        Column('card_id', String, ForeignKey(Card.id), index=True),
        Column('deck_id', String),
        Column('label', Boolean),
        Column('date', TIMESTAMP(timezone=True)),
        Column('elapsed_milliseconds_text', Integer),
        Column('elapsed_milliseconds_answer', Integer),
        Column('times_seen', Integer, nullable=False, default=0),
        Column('times_seen_in_session', Integer, nullable=False, default=0),
    )

    meta.create_all(engine)

    session = SessionLocal()
    records = session.query(Record).order_by(Record.date)
    study_records_to_insert = []
    times_seen = dict()  # times_seen[user_id][card_id]
    front_end_ids = set()  # should be unique
    for record in tqdm(records, total=records.count()):
        if record.user_id not in times_seen:
            times_seen[record.user_id] = dict()
        if record.card_id not in times_seen[record.user_id]:
            times_seen[record.user_id][record.card_id] = 0

        if record.front_end_id in front_end_ids:
            continue
        else:
            front_end_ids.add(record.front_end_id)

        study_records_to_insert.append(
            StudyRecord(
                id=record.front_end_id,
                debug_id=record.id,  # aka debug_id
                studyset_id=record.id,  # using debug_id here since existing records are created without sessions, in other words each study is a single session
                user_id=record.user_id,
                card_id=record.card_id,
                deck_id=record.deck_id,
                label=record.response,
                date=record.date,
                elapsed_milliseconds_text=record.elapsed_milliseconds_text,
                elapsed_milliseconds_answer=record.elapsed_milliseconds_answer,
                times_seen=times_seen[record.user_id][record.card_id],
                times_seen_in_session=0,  # single study session
            )
        )
        times_seen[record.user_id][record.card_id] += 1
    session.bulk_save_objects(study_records_to_insert)
    session.commit()
    session.close()


def downgrade():
    StudyRecord.__table__.drop(engine)
