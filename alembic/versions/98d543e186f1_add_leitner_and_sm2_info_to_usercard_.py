"""add leitner and sm2 info to usercard feature vector

Revision ID: 98d543e186f1
Revises: da925c768b0b
Create Date: 2020-12-20 18:47:49.891642

"""
from tqdm import tqdm
from datetime import datetime, timedelta
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, Integer, TIMESTAMP

from karl.models import User, Record, UserCardFeatureVector


# revision identifiers, used by Alembic.
revision = '98d543e186f1'
down_revision = 'da925c768b0b'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.add_column(
        'usercardfeaturevector',
        Column('leitner_box', Integer),
    )
    op.add_column(
        'usercardfeaturevector',
        Column('leitner_scheduled_date', TIMESTAMP(timezone=True)),
    )
    op.add_column(
        'usercardfeaturevector',
        Column('sm2_scheduled_date', TIMESTAMP(timezone=True)),
    )


def schema_downgrade():
    op.drop_column('usercardfeaturevector', 'leitner_box')
    op.drop_column('usercardfeaturevector', 'leitner_scheduled_date')
    op.drop_column('usercardfeaturevector', 'sm2_scheduled_date')


def update_leitner(
    record: Record,
    date: datetime,
    leitner_box: dict,
    leitner_scheduled_date: dict,
):
    # leitner boxes 1~10
    # days[0] = None as placeholder since we don't have box 0
    # days[9] and days[10] = 999 to make it never repeat
    days = [0, 0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 999, 999]
    increment_days = {i: x for i, x in enumerate(days)}

    if record.card_id not in leitner_box:
        # boxes: 1 ~ 10
        leitner_box[record.card_id] = 1

    leitner_box[record.card_id] += (1 if record.response else -1)
    leitner_box[record.card_id] = max(min(leitner_box[record.card_id], 10), 1)
    interval = timedelta(days=increment_days[leitner_box[record.card_id]])
    leitner_scheduled_date[record.card_id] = date + interval
    return leitner_box, leitner_scheduled_date

def update_sm2(
    record: Record,
    date: datetime,
    sm2_efactor: dict,
    sm2_interval: dict,
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
    return sm2_efactor, sm2_interval, sm2_repetition, sm2_scheduled_date


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session.query(User)
    for user in tqdm(users, total=users.count()):
        leitner_box, leitner_scheduled_date = {}, {}
        sm2_interval, sm2_efactor, sm2_repetition, sm2_scheduled_date = {}, {}, {}, {}
        for record in user.records:
            usercard_feature_vector = session.query(UserCardFeatureVector).get(record.id)
            if usercard_feature_vector is None:
                print(user.id, record.id)
                continue
            usercard_feature_vector.leitner_box = leitner_box.get(record.card_id, None)
            usercard_feature_vector.leitner_scheduled_date = leitner_scheduled_date.get(record.card_id, None)
            usercard_feature_vector.sm2_scheduled_date = sm2_scheduled_date.get(record.card_id, None)

            sm2_efactor, sm2_interval, sm2_reptition, sm2_scheduled_date = update_sm2(record, record.date, sm2_efactor, sm2_interval, sm2_repetition, sm2_scheduled_date)
            leitner_box, leitner_scheduled_date = update_leitner(record, record.date, leitner_box, leitner_scheduled_date)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
