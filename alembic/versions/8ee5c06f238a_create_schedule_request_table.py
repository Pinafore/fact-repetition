"""create schedule request table

Revision ID: 8ee5c06f238a
Revises: 379e01a6c6a4
Create Date: 2020-12-13 01:41:19.647204

"""
import json
import pytz
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, String, ARRAY, TIMESTAMP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import Record as OldRecord
from karl.models import Record, ScheduleRequest
from karl.config import settings


# revision identifiers, used by Alembic.
revision = '8ee5c06f238a'
down_revision = '379e01a6c6a4'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'schedulerequest',
        Column('id', String, ForeignKey(Record.id), primary_key=True, index=True),
        Column('card_ids', ARRAY(String), nullable=False),
        Column('date', TIMESTAMP(timezone=True)),
    )


def schema_downgrade():
    op.drop_table('schedulerequest')


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

        if session.query(Record).get(new_id) is None:
            continue

        request = ScheduleRequest(
            id=new_id,
            card_ids=record_old.fact_ids,
        )

        session.add(request)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
