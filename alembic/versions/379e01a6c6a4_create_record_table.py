"""create record table

Revision ID: 379e01a6c6a4
Revises: f095364f3c69
Create Date: 2020-12-13 01:19:35.103848

"""
import json
import pytz
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, String, Integer, Boolean, TIMESTAMP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import Record as OldRecord
from karl.models import User, Card, Record
from karl.config import settings


# revision identifiers, used by Alembic.
revision = '379e01a6c6a4'
down_revision = 'f095364f3c69'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'record',
        Column('id', String, primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id), index=True),
        Column('card_id', String, ForeignKey(Card.id), index=True),
        Column('front_end_id', String, index=True),
        Column('deck_id', String),
        Column('response', Boolean),
        Column('elapsed_milliseconds_text', Integer),
        Column('elapsed_milliseconds_answer', Integer),
        Column('is_new_fact', Boolean, nullable=False),
        Column('date', TIMESTAMP(timezone=True)),
    )


def schema_downgrade():
    op.drop_table('record')


def data_upgrade():
    engine = create_engine(settings.STABLE_DATABASE_URL)
    session_remote = sessionmaker(bind=engine, autoflush=False)()

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    keys = set()
    records = session_remote.query(OldRecord)
    for record_old in tqdm(records, total=records.count()):
        if record_old.response is None:
            continue

        if not record_old.user_id.isdigit():
            continue

        new_id = json.dumps({
            'user_id': record_old.user_id,
            'card_id': record_old.fact_id,
            'date': str(record_old.date.replace(tzinfo=pytz.UTC)),
        })
        if new_id in keys:
            continue
        else:
            keys.add(new_id)

        record_new = Record(
            id=new_id,
            user_id=record_old.user_id,
            card_id=record_old.fact_id,
            front_end_id=record_old.record_id,
            deck_id=record_old.__dict__.get('deck_id', None),
            response=record_old.response,
            elapsed_milliseconds_text=record_old.elapsed_milliseconds_text,
            elapsed_milliseconds_answer=record_old.elapsed_milliseconds_answer,
            is_new_fact=record_old.is_new_fact,
            date=record_old.date.replace(tzinfo=pytz.UTC),
        )

        session.add(record_new)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
