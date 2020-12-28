"""add more sm2 info to usercard feature vector

Revision ID: 75b10cc58867
Revises: 98d543e186f1
Create Date: 2020-12-28 15:31:10.281209

"""
from tqdm import tqdm
from datetime import datetime, timedelta
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, Integer, Float

from karl.models import User, Record, UserCardFeatureVector


# revision identifiers, used by Alembic.
revision = '75b10cc58867'
down_revision = '98d543e186f1'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.add_column(
        'usercardfeaturevector',
        Column('sm2_efactor', Float),
    )
    op.add_column(
        'usercardfeaturevector',
        Column('sm2_interval', Float),
    )
    op.add_column(
        'usercardfeaturevector',
        Column('sm2_repetition', Integer),
    )


def schema_downgrade():
    op.drop_column('usercardfeaturevector', 'sm2_efactor')
    op.drop_column('usercardfeaturevector', 'sm2_interval')
    op.drop_column('usercardfeaturevector', 'sm2_repetition')


def update_sm2(
    record: Record,
    date: datetime,
    sm2_interval: dict,
    sm2_efactor: dict,
    sm2_repetition: dict,
    sm2_scheduled_date: dict,
) -> None:
    def get_quality_from_response(response: bool) -> int:
        return 4 if response else 1

    if record.card_id not in sm2_scheduled_date:
        sm2_efactor[record.card_id] = 2.5
        sm2_interval[record.card_id] = 1
        sm2_repetition[record.card_id] = 0

    q = get_quality_from_response(record.response)
    sm2_repetition[record.card_id] += 1
    sm2_efactor[record.card_id] = max(1.3, sm2_efactor[record.card_id] + 0.1 - (5.0 - q) * (0.08 + (5.0 - q) * 0.02))

    if not record.response:
        sm2_interval[record.card_id] = 0
        sm2_repetition[record.card_id] = 0
    else:
        if sm2_repetition[record.card_id] == 1:
            sm2_interval[record.card_id] = 1
        elif sm2_repetition[record.card_id] == 2:
            sm2_interval[record.card_id] = 6
        else:
            sm2_interval[record.card_id] *= sm2_efactor[record.card_id]

    sm2_scheduled_date[record.card_id] = date + timedelta(days=sm2_interval[record.card_id])
    # return sm2_interval, sm2_efactor, sm2_repetition, sm2_scheduled_date


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session.query(User)
    for user in tqdm(users, total=users.count()):
        sm2_interval, sm2_efactor, sm2_repetition, sm2_scheduled_date = {}, {}, {}, {}
        for record in user.records:
            usercard_feature_vector = session.query(UserCardFeatureVector).get(record.id)
            if usercard_feature_vector is None:
                print(user.id, record.id)
                continue
            usercard_feature_vector.sm2_efactor = sm2_efactor.get(record.card_id, None)
            usercard_feature_vector.sm2_interval = sm2_interval.get(record.card_id, None)
            usercard_feature_vector.sm2_repetition = sm2_repetition.get(record.card_id, None)

            update_sm2(record, record.date, sm2_interval, sm2_efactor, sm2_repetition, sm2_scheduled_date)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
