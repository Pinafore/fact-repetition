"""recover scheduler output

Revision ID: e2b7e34d9144
Revises: be1da3c4f8a6
Create Date: 2020-10-28 02:37:35.375600

"""
import sqlalchemy as sa
from alembic import op
from alembic import context
from sqlalchemy import orm
from sqlalchemy.ext.mutable import MutableDict
from tqdm import tqdm

from karl.models import JSONEncoded, Record, SchedulerOutput
from karl.scheduler import MovingAvgScheduler


# revision identifiers, used by Alembic.
revision = 'e2b7e34d9144'
down_revision = 'be1da3c4f8a6'
branch_labels = None
depends_on = None


def schema_upgrades():
    op.create_table(
        'scheduler_outputs',
        sa.Column('debug_id', sa.String, primary_key=True),
        sa.Column('order', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('scores', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('details', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('rationale', sa.String),
        sa.Column('record_id', sa.String, sa.ForeignKey('records.record_id')),
    )


def data_upgrades():
    bind = op.get_bind()
    session = orm.Session(bind=bind)
    scheduler = MovingAvgScheduler(preemptive=False)

    records = session.query(Record).order_by(Record.date)
    for i, record in enumerate(tqdm(records, total=records.count())):
        # expunge user and record so they can be restored withouth affecting the value in DB
        user = record.user
        user_records = user.records
        if not sa.inspect(user).detached:
            session.expunge(user)
        user.records = user_records

        fact = record.fact
        if not sa.inspect(fact).detached:
            session.expunge(fact)

        fact.results = record.fact_snapshot.results.get(fact.fact_id, [])

        user.previous_study = record.user_snapshot.previous_study
        user.results = record.user_snapshot.results
        user.count_correct_before = record.user_snapshot.count_correct_before
        user.count_wrong_before = record.user_snapshot.count_wrong_before
        user.params = record.user_snapshot.params

        scheduler_output = scheduler.rank_facts_for_user(user, [fact], record.date)
        scheduler_output_in_db = SchedulerOutput(
            debug_id=record.debug_id,
            order=scheduler_output.order,
            scores=scheduler_output.scores,
            details=scheduler_output.details,
            rationale=scheduler_output.rationale,
            record_id=record.record_id,
        )
        session.add(scheduler_output_in_db)

        if i % 2000 == 0 and i > 0:
            session.commit()

    session.commit()


def schema_downgrades():
    op.drop_table('scheduler_outputs')


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
