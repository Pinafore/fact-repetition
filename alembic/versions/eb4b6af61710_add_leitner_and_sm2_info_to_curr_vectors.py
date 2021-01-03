"""add leitner and sm2 info to curr vectors

Revision ID: eb4b6af61710
Revises: b2997f1dca3e
Create Date: 2021-01-03 03:38:36.952385

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, Integer, Float, TIMESTAMP

from karl.models import CurrUserCardFeatureVector, Leitner, SM2


# revision identifiers, used by Alembic.
revision = 'eb4b6af61710'
down_revision = 'b2997f1dca3e'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.add_column(
        'currusercardfeaturevector',
        Column('leitner_box', Integer),
    )
    op.add_column(
        'currusercardfeaturevector',
        Column('leitner_scheduled_date', TIMESTAMP(timezone=True)),
    )
    op.add_column(
        'currusercardfeaturevector',
        Column('sm2_efactor', Float),
    )
    op.add_column(
        'currusercardfeaturevector',
        Column('sm2_interval', Float),
    )
    op.add_column(
        'currusercardfeaturevector',
        Column('sm2_repetition', Integer),
    )
    op.add_column(
        'currusercardfeaturevector',
        Column('sm2_scheduled_date', TIMESTAMP(timezone=True)),
    )


def schema_downgrade():
    op.drop_column('currusercardfeaturevector', 'leitner_box')
    op.drop_column('currusercardfeaturevector', 'leitner_scheduled_date')
    op.drop_column('currusercardfeaturevector', 'sm2_efactor')
    op.drop_column('currusercardfeaturevector', 'sm2_interval')
    op.drop_column('currusercardfeaturevector', 'sm2_repetition')
    op.drop_column('currusercardfeaturevector', 'sm2_scheduled_date')


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    vectors = session.query(CurrUserCardFeatureVector)
    for v_usercard in tqdm(vectors, total=vectors.count()):
        leitner = session.query(Leitner).get((v_usercard.user_id, v_usercard.card_id))
        sm2 = session.query(SM2).get((v_usercard.user_id, v_usercard.card_id))
        if leitner is not None:
            v_usercard.leitner_box = leitner.box
            v_usercard.leitner_scheduled_date = leitner.scheduled_date
        if sm2 is not None:
            v_usercard.sm2_efactor = sm2.efactor
            v_usercard.sm2_interval = sm2.interval
            v_usercard.sm2_repetition = sm2.repetition
            v_usercard.sm2_scheduled_date = sm2.scheduled_date

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
