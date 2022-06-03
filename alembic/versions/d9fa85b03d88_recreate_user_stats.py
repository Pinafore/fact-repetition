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
from tqdm import tqdm
from karl.db.session import SessionLocal, engine
from karl.models import StudyRecord, User, UserStatsV2
from karl.scheduler import KARLScheduler


# revision identifiers, used by Alembic.
revision = 'd9fa85b03d88'
down_revision = 'b179baed0aef'
branch_labels = None
depends_on = None


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
    scheduler = KARLScheduler()

    records = session.query(StudyRecord).order_by(StudyRecord.date)
    for record in tqdm(records, total=records.count()):
        utc_date = record.date.astimezone(pytz.utc).date()
        scheduler.update_user_stats(record, deck_id='all', utc_date=utc_date, session=session)
        if record.deck_id is not None:
            scheduler.update_user_stats(record, deck_id=record.deck_id, utc_date=utc_date, session=session)
        session.commit()


def downgrade():
    pass
