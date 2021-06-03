"""reproduce feature vectors

Revision ID: 816a8cde12b6
Revises: c3fb02a02e58
Create Date: 2021-03-23 21:08:48.959181

"""
from tqdm import tqdm
from sqlalchemy import Column, ForeignKey, Integer, Float, Boolean, String, TIMESTAMP
from sqlalchemy import Table, MetaData
from sqlalchemy.dialects.postgresql import JSONB

from karl.models import CurrCardFeatureVector, CurrUserFeatureVector, CurrUserCardFeatureVector,\
    UserFeatureVector, CardFeatureVector, UserCardFeatureVector
from karl.scheduler import KARLScheduler
from karl.models import User, Card, Record
from karl.db.session import SessionLocal, engine


# revision identifiers, used by Alembic.
revision = '816a8cde12b6'
down_revision = None
branch_labels = None
depends_on = None


def data_upgrade():
    scheduler = KARLScheduler()
    # bind = op.get_bind()
    # session = orm.Session(bind=bind)
    session = SessionLocal()
    records = session.query(Record).order_by(Record.date)
    session.close()
    for record in tqdm(records, total=records.count()):
        # If a current feature vector exists (for user, card, or user-card),
        # that's the input to the retention model to create `record`,
        # and we store that feature vector with `record.id`.
        #
        # If not, we initialize everything with None,
        # and that's the input we store with `record.id`.
        # Then we update the None feature vector with response from this record,
        # which becomes the new current feature vector for the next round.
        if record.response is None:
            continue

        if not record.user_id.isdigit():
            # dummy users for development
            continue

        # simulate scheduling request
        scheduler.save_feature_vectors(record.id, record.user_id, record.card_id, record.date)

        # simulate update request
        session = SessionLocal()
        scheduler.update_feature_vectors(record, record.date, session)
        session.commit()
        session.close()


def schema_upgrade():
    meta = MetaData()

    Table(
        'usercardfeaturevector', meta,
        Column('id', String, ForeignKey(Record.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id, ondelete='CASCADE'), index=True),
        Column('card_id', String, ForeignKey(Card.id, ondelete='CASCADE'), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('leitner_box', Integer),
        Column('leitner_scheduled_date', TIMESTAMP(timezone=True)),
        Column('sm2_efactor', Float),
        Column('sm2_interval', Float),
        Column('sm2_repetition', Integer),
        Column('sm2_scheduled_date', TIMESTAMP(timezone=True)),
        Column('correct_on_first_try', Boolean),
    )

    Table(
        'userfeaturevector', meta,
        Column('id', String, ForeignKey(Record.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id, ondelete='CASCADE'), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('parameters', JSONB),
    )

    Table(
        'cardfeaturevector', meta,
        Column('id', String, ForeignKey(Record.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id, ondelete='CASCADE'), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    Table(
        'currusercardfeaturevector', meta,
        Column('user_id', String, ForeignKey(User.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('correct_on_first_try', Boolean),
        Column('leitner_box', Integer),
        Column('leitner_scheduled_date', TIMESTAMP(timezone=True)),
        Column('sm2_efactor', Float),
        Column('sm2_interval', Float),
        Column('sm2_repetition', Integer),
        Column('sm2_scheduled_date', TIMESTAMP(timezone=True)),
    )

    Table(
        'curruserfeaturevector', meta,
        Column('user_id', String, ForeignKey(User.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
        Column('parameters', JSONB),
    )

    Table(
        'currcardfeaturevector', meta,
        Column('card_id', String, ForeignKey(Card.id, ondelete='CASCADE'), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    meta.create_all(engine)


def upgrade():
    UserCardFeatureVector.__table__.drop(engine)
    UserFeatureVector.__table__.drop(engine)
    CardFeatureVector.__table__.drop(engine)
    CurrUserCardFeatureVector.__table__.drop(engine)
    CurrUserFeatureVector.__table__.drop(engine)
    CurrCardFeatureVector.__table__.drop(engine)

    schema_upgrade()
    data_upgrade()


def downgrade():
    pass


upgrade()
