"""recreate schedule request table

Revision ID: c63c0f83d6da
Revises: 
Create Date: 2022-06-01 15:56:50.389448

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Table, MetaData
from sqlalchemy import Column, ForeignKey, ARRAY, String, TIMESTAMP
from tqdm import tqdm

from karl.models import Record, ScheduleRequest
from karl.db.session import SessionLocal, engine


# revision identifiers, used by Alembic.
revision = 'c63c0f83d6da'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    ScheduleRequest.__table__.drop(engine)

    meta = MetaData()

    Table(
        'schedulerequest', meta,
        Column('id', String, primary_key=True, index=True),
        Column('card_ids', ARRAY(String), default=[]),
        Column('date', TIMESTAMP(timezone=True)),
    )

    meta.create_all(engine)

    session = SessionLocal()
    records = session.query(Record).order_by(Record.date)
    schedule_requests_to_insert = []
    for record in tqdm(records, total=records.count()):
        schedule_requests_to_insert.append(
            ScheduleRequest(
                id=record.id,
                card_ids=[],
                date=record.date,
            )
        )
    session.bulk_save_objects(schedule_requests_to_insert)
    session.commit()
    session.close()
    

def downgrade():
    pass
