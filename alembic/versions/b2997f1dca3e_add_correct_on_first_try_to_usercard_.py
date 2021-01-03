"""add correct_on_first_try to usercard vectors

Revision ID: b2997f1dca3e
Revises: 98d543e186f1
Create Date: 2020-12-29 00:33:25.319189

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, Boolean

from karl.models import User, UserCardFeatureVector, CurrUserCardFeatureVector


# revision identifiers, used by Alembic.
revision = 'b2997f1dca3e'
down_revision = '98d543e186f1'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.add_column(
        'usercardfeaturevector',
        Column('correct_on_first_try', Boolean),
    )
    op.add_column(
        'currusercardfeaturevector',
        Column('correct_on_first_try', Boolean),
    )


def schema_downgrade():
    op.drop_column('usercardfeaturevector', 'correct_on_first_try')
    op.drop_column('currusercardfeaturevector', 'correct_on_first_try')


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session.query(User)
    for user in tqdm(users, total=users.count()):
        correct_on_first_try = {}  # card_id -> boolean
        for record in user.records:
            usercard_feature_vector = session.query(UserCardFeatureVector).get(record.id)
            if usercard_feature_vector is None:
                print(user.id, record.id)
                continue
            if record.card_id not in correct_on_first_try:
                correct_on_first_try[record.card_id] = record.response
            else:
                # usercard_feature_vector is what's visible during the scheduling of this record
                # before the response is given by the user.
                # so in the first occurrence, `correct_on_first_try` should be left None
                usercard_feature_vector.correct_on_first_try = correct_on_first_try[record.card_id]
        for card_id, response in correct_on_first_try.items():
            curr_usercard_vector = session.query(CurrUserCardFeatureVector).get((user.id, card_id))
            curr_usercard_vector.correct_on_first_try = response

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
