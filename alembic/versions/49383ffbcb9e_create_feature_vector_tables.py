"""create feature vector tables

Revision ID: 49383ffbcb9e
Revises: 825204dece93
Create Date: 2020-12-13 20:44:03.060891

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, Integer, Boolean, String, TIMESTAMP

from karl.models import User, Card, Record
from karl.models import UserCardFeatureVector, UserFeatureVector, CardFeatureVector
from karl.models import CurrUserCardFeatureVector, CurrUserFeatureVector, CurrCardFeatureVector


# revision identifiers, used by Alembic.
revision = '49383ffbcb9e'
down_revision = '825204dece93'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'usercardfeaturevector',
        Column('id', String, ForeignKey(Record.id), primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id), index=True),
        Column('card_id', String, ForeignKey(Card.id), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    op.create_table(
        'userfeaturevector',
        Column('id', String, ForeignKey(Record.id), primary_key=True, index=True),
        Column('user_id', String, ForeignKey(User.id), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    op.create_table(
        'cardfeaturevector',
        Column('id', String, ForeignKey(Record.id), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id), index=True),
        Column('date', TIMESTAMP(timezone=True), index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('delta', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    op.create_table(
        'currusercardfeaturevector',
        Column('user_id', String, ForeignKey(User.id), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    op.create_table(
        'curruserfeaturevector',
        Column('user_id', String, ForeignKey(User.id), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )

    op.create_table(
        'currcardfeaturevector',
        Column('card_id', String, ForeignKey(Card.id), primary_key=True, index=True),
        Column('n_study_positive', Integer),
        Column('n_study_negative', Integer),
        Column('n_study_total', Integer),
        Column('previous_delta', Integer),
        Column('previous_study_date', TIMESTAMP(timezone=True)),
        Column('previous_study_response', Boolean),
    )


def schema_downgrade():
    op.drop_table('usercardfeaturevector')
    op.drop_table('userfeaturevector')
    op.drop_table('cardfeaturevector')
    op.drop_table('currusercardfeaturevector')
    op.drop_table('curruserfeaturevector')
    op.drop_table('currcardfeaturevector')


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    records = session.query(Record).order_by(Record.date)
    for record in tqdm(records, total=records.count()):
        # If a current feature vector exists (for user, card, or user-card),
        # that's the input to the retention model to create `record`,
        # and we store that feature vector with `record.id`.
        #
        # If not, we initialize everything with None,
        # and that's the input we store with `record.id`.
        # Then we update the None feature vector with response from this record,
        # which becomes the new current feature vector for the next round.

        curr_usercard_vector = session.query(CurrUserCardFeatureVector).get((record.user_id, record.card_id))
        if curr_usercard_vector is None:
            curr_usercard_vector = CurrUserCardFeatureVector(
                user_id=record.user_id,
                card_id=record.card_id,
                n_study_positive=0,
                n_study_negative=0,
                n_study_total=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
            )
            session.add(curr_usercard_vector)

        curr_user_vector = session.query(CurrUserFeatureVector).get(record.user_id)
        if curr_user_vector is None:
            curr_user_vector = CurrUserFeatureVector(
                user_id=record.user_id,
                n_study_positive=0,
                n_study_negative=0,
                n_study_total=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
            )
            session.add(curr_user_vector)

        curr_card_vector = session.query(CurrCardFeatureVector).get(record.card_id)
        if curr_card_vector is None:
            curr_card_vector = CurrCardFeatureVector(
                card_id=record.card_id,
                n_study_positive=0,
                n_study_negative=0,
                n_study_total=0,
                previous_delta=None,
                previous_study_date=None,
                previous_study_response=None,
            )
            session.add(curr_card_vector)

        # save the input to the retention model for generating the current record
        usercard_delta = None
        if curr_usercard_vector.previous_study_date is not None:
            usercard_delta = (record.date - curr_usercard_vector.previous_study_date).total_seconds()
        usercard_vector = UserCardFeatureVector(
            id=record.id,
            user_id=record.user_id,
            card_id=record.card_id,
            date=record.date,
            n_study_positive=curr_usercard_vector.n_study_positive,
            n_study_negative=curr_usercard_vector.n_study_negative,
            n_study_total=curr_usercard_vector.n_study_total,
            delta=usercard_delta,
            previous_delta=curr_usercard_vector.previous_delta,
            previous_study_date=curr_usercard_vector.previous_study_date,
            previous_study_response=curr_usercard_vector.previous_study_response,
        )
        session.add(usercard_vector)

        user_delta = None
        if curr_user_vector.previous_study_date is not None:
            user_delta = (record.date - curr_user_vector.previous_study_date).total_seconds()
        user_vector = UserFeatureVector(
            id=record.id,
            user_id=record.user_id,
            date=record.date,
            n_study_positive=curr_user_vector.n_study_positive,
            n_study_negative=curr_user_vector.n_study_negative,
            n_study_total=curr_user_vector.n_study_total,
            delta=user_delta,
            previous_delta=curr_user_vector.previous_delta,
            previous_study_date=curr_user_vector.previous_study_date,
            previous_study_response=curr_user_vector.previous_study_response,
        )
        session.add(user_vector)

        card_delta = None
        if curr_card_vector.previous_study_date is not None:
            card_delta = (record.date - curr_card_vector.previous_study_date).total_seconds()
        card_vector = CardFeatureVector(
            id=record.id,
            card_id=record.card_id,
            date=record.date,
            n_study_positive=curr_card_vector.n_study_positive,
            n_study_negative=curr_card_vector.n_study_negative,
            n_study_total=curr_card_vector.n_study_total,
            delta=card_delta,
            previous_delta=curr_card_vector.previous_delta,
            previous_study_date=curr_card_vector.previous_study_date,
            previous_study_response=curr_card_vector.previous_study_response,
        )
        session.add(card_vector)

        # update curr feature vectors
        curr_usercard_vector.n_study_positive += record.response
        curr_usercard_vector.n_study_negative += not record.response
        curr_usercard_vector.n_study_total += 1
        curr_usercard_vector.previous_delta = usercard_delta
        curr_usercard_vector.previous_study_date = record.date
        curr_usercard_vector.previous_study_response = record.response

        curr_user_vector.n_study_positive += record.response
        curr_user_vector.n_study_negative += not record.response
        curr_user_vector.n_study_total += 1
        curr_user_vector.previous_delta = user_delta
        curr_user_vector.previous_study_date = record.date
        curr_user_vector.previous_study_response = record.response

        curr_card_vector.n_study_positive += record.response
        curr_card_vector.n_study_negative += not record.response
        curr_card_vector.n_study_total += 1
        curr_card_vector.previous_delta = card_delta
        curr_card_vector.previous_study_date = record.date
        curr_card_vector.previous_study_response = record.response

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
