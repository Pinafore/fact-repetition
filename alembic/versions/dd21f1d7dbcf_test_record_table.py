"""test record table

Revision ID: dd21f1d7dbcf
Revises: 3dd92c6f4fb4
Create Date: 2022-06-01 22:23:39.616989

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import Table, MetaData
from sqlalchemy import Column, ForeignKey, ARRAY, String, TIMESTAMP, Boolean, Integer

from karl.models import User, Card
from karl.db.session import SessionLocal, engine

# revision identifiers, used by Alembic.
revision = 'dd21f1d7dbcf'
down_revision = '3dd92c6f4fb4'
branch_labels = None
depends_on = None


def upgrade():
    meta = MetaData()

    Table(
        'testrecord', meta,
        Column('id', String, primary_key=True, index=True),  # history_id / front_end_id provided by Matthew
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


def downgrade():
    pass
