"""add request snapshot to record

Revision ID: be1da3c4f8a6
Revises: 8f756504c67f
Create Date: 2020-10-25 23:55:54.494389

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Date
from sqlalchemy.ext.mutable import MutableDict, MutableList
from sqlalchemy.ext.declarative import declarative_base

from tqdm import tqdm
from karl.models import Record, User, UserSnapshot 
from karl.scheduler import MovingAvgScheduler
from karl.models import JSONEncoded, ParamsType


# revision identifiers, used by Alembic.
revision = 'be1da3c4f8a6'
down_revision = '8f756504c67f'
branch_labels = None
depends_on = None


Base = declarative_base()


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    # scheduler = MovingAvgScheduler(preemptive=False)

    op.create_table(
        'user_snapshots',
        Column('debug_id', String),
        Column('user_id', String),
        Column('record_id', String),
        Column('date', DateTime),
        Column('recent_facts', MutableList.as_mutable(JSONEncoded)),
        Column('previous_study', MutableDict.as_mutable(JSONEncoded)),
        Column('leitner_box', MutableDict.as_mutable(JSONEncoded)),
        Column('leitner_scheduled_date', MutableDict.as_mutable(JSONEncoded)),
        Column('sm2_efactor', MutableDict.as_mutable(JSONEncoded)),
        Column('sm2_interval', MutableDict.as_mutable(JSONEncoded)),
        Column('sm2_repetition', MutableDict.as_mutable(JSONEncoded)),
        Column('sm2_scheduled_date', MutableDict.as_mutable(JSONEncoded)),
        Column('results', MutableList.as_mutable(JSONEncoded)),
        Column('count_correct_before', MutableDict.as_mutable(JSONEncoded)),
        Column('count_wrong_before', MutableDict.as_mutable(JSONEncoded)),
        Column('params', ParamsType),
        sa.PrimaryKeyConstraint('debug_id')
    )

    op.create_foreign_key(
        constraint_name='user_snapshot_user_id',
        source_table='user_snapshots',
        referent_table='users',
        local_cols=['user_id'],
        remote_cols=['user_id']
    )

    op.create_foreign_key(
        constraint_name='user_snapshot_record_id',
        source_table='user_snapshots',
        referent_table='records',
        local_cols=['record_id'],
        remote_cols=['record_id']
    )

    op.add_column('records', sa.Column('new_user_snapshot', sa.String))
    op.add_column('records', sa.Column('new_fact_snapshot', sa.String))
    op.add_column('records', sa.Column('new_scheduler_output', sa.String))

    # op.drop_column('records', 'user_snapshot')
    # op.drop_column('records', 'scheduler_output')
    # op.drop_column('records', 'scheduler_snapshot')

    # op.alter_column('records', 'new_user_snapshot', new_column_name='user_snapshot')
    # op.alter_column('records', 'new_fact_snapshot', new_column_name='fact_snapshot')
    # op.alter_column('records', 'new_scheduler_output', new_column_name='scheduler_output')

    debug_id = 'test_debug_id'

    user = session.query(User).get('463')
    record = user.records[-1]
    user_snapshot = UserSnapshot(
        debug_id=debug_id,
        user_id=user.user_id,
        record_id=record.record_id,
        date=record.date,
        recent_facts=[record.fact_id for record in user.records[::-1][:user.params.max_recent_facts]],
        previous_study=user.previous_study,
        leitner_box=user.leitner_box,
        leitner_scheduled_date={k: str(v) for k, v in user.leitner_scheduled_date},
        sm2_efactor=user.sm2_efactor,
        sm2_interval=user.sm2_interval,
        sm2_repetition=user.sm2_repetition,
        sm2_scheduled_date={k: str(v) for k, v in user.sm2_scheduled_date},
        results=user.results,
        count_correct_before=user.count_correct_before,
        count_wrong_before=user.count_wrong_before,
        params=user.params,
    )
    session.add(user_snapshot)
    session.commit()

    user_snapshot = session.query(UserSnapshot).get(debug_id)
    print(user_snapshot.leitner_box)
    print(user_snapshot.recent_facts)


def downgrade():
    pass
