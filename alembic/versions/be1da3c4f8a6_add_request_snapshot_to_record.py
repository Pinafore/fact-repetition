"""add request snapshot to record

Revision ID: be1da3c4f8a6
Revises: 8f756504c67f
Create Date: 2020-10-25 23:55:54.494389

"""
import json
import hashlib
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy.ext.mutable import MutableDict, MutableList
from alembic import context

from tqdm import tqdm
from karl.models import Record, UserSnapshot, FactSnapshot
from karl.util import Params
from karl.models import JSONEncoded, ParamsType


# revision identifiers, used by Alembic.
revision = 'be1da3c4f8a6'
down_revision = '8f756504c67f'
branch_labels = None
depends_on = None


def schema_upgrades():
    op.create_table(
        'user_snapshots',
        sa.Column('debug_id', sa.String, primary_key=True),
        sa.Column('record_id', sa.String, sa.ForeignKey('records.record_id')),
        sa.Column('user_id', sa.String, sa.ForeignKey('users.user_id')),
        sa.Column('date', sa.DateTime),
        sa.Column('recent_facts', MutableList.as_mutable(JSONEncoded)),
        sa.Column('previous_study', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('leitner_box', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('leitner_scheduled_date', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_efactor', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_interval', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_repetition', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_scheduled_date', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('results', MutableList.as_mutable(JSONEncoded)),
        sa.Column('count_correct_before', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('count_wrong_before', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('params', ParamsType),
    )

    op.create_table(
        'fact_snapshots',
        sa.Column('id', sa.Integer, primary_key=True, autoincrement=True),
        sa.Column('results', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('record_id', sa.String, sa.ForeignKey('records.record_id')),
    )


def data_upgrades():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    # below needs to be manually updated after scheduler is called
    all_fact_results = {}  # fact_id -> list of binary results
    user_seen_facts = {}  # user_id -> list of fact_ids from old to new

    records = session.query(Record).order_by(Record.date)
    for i, record in enumerate(tqdm(records, total=records.count())):
        # expunge user and record so they can be restored withouth affecting the value in DB
        user = record.user
        if not sa.inspect(user).detached:
            session.expunge(user)

        fact = record.fact
        if not sa.inspect(fact).detached:
            session.expunge(fact)

        # if record.debug_id == 'null':
        record.debug_id = hashlib.md5(
            json.dumps({
                'user_id': record.user_id,
                'fact_id': record.fact_id,
                'date': record.date.strftime('%Y-%m-%dT%H:%M:%S%z'),
            }).encode('utf8')).hexdigest()

        snapshot = json.loads(record.user_snapshot)

        # restore user
        user.previous_study = snapshot['previous_study']
        user.results = snapshot['user_results']
        user.count_correct_before = snapshot['count_correct_before']
        user.count_wrong_before = snapshot['count_wrong_before']
        user.params = Params(**json.loads(record.scheduler_snapshot))

        user_snapshot = UserSnapshot(
            debug_id=record.debug_id,
            user_id=user.user_id,
            record_id=record.record_id,
            date=record.date,
            recent_facts=user_seen_facts.get(user.user_id, [])[::-1][:user.params.max_recent_facts],
            previous_study=user.previous_study,
            leitner_box=user.leitner_box,
            leitner_scheduled_date={k: str(v) for k, v in user.leitner_scheduled_date.items()},  # TODO
            sm2_efactor=user.sm2_efactor,  # TODO
            sm2_interval=user.sm2_interval,  # TODO
            sm2_repetition=user.sm2_repetition,  # TODO
            sm2_scheduled_date={k: str(v) for k, v in user.sm2_scheduled_date.items()},  # TODO
            results=user.results,
            count_correct_before=user.count_correct_before,
            count_wrong_before=user.count_wrong_before,
            params=user.params,
        )

        fact_snapshot = FactSnapshot(
            record_id=record.record_id,
            results=all_fact_results
        )

        session.add(user_snapshot)
        session.add(fact_snapshot)

        # update all_fact_results
        if record.fact_id not in all_fact_results:
            all_fact_results[record.fact_id] = []
        all_fact_results[record.fact_id].append(record.response)

        # update user_seen_facts
        if user.user_id not in user_seen_facts:
            user_seen_facts[user.user_id] = []
        user_seen_facts[user.user_id].append(record.fact_id)

        if i % 2000 == 0 and i > 0:
            session.commit()

    session.commit()


def schema_downgrades():
    op.drop_table('user_snapshots')
    op.drop_table('fact_snapshots')


def upgrade():
    schema_upgrades()
    if context.get_x_argument(as_dictionary=True).get('data', False):
        data_upgrades()


def downgrade():
    if context.get_x_argument(as_dictionary=True).get('data', False):
        data_downgrades()
    schema_downgrades()


def data_downgrades():
    pass
