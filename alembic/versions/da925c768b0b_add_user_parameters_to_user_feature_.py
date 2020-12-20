"""add user parameters to user feature vector

Revision ID: da925c768b0b
Revises: 49383ffbcb9e
Create Date: 2020-12-20 18:18:50.738873

"""
import json
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB

from karl.models import User, UserFeatureVector
from karl.schemas import ParametersSchema


# revision identifiers, used by Alembic.
revision = 'da925c768b0b'
down_revision = '49383ffbcb9e'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.add_column(
        'userfeaturevector',
        Column('parameters', JSONB),
    )


def schema_downgrade():
    op.drop_column('userfeaturevector', 'parameters')


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session.query(User)
    for user in tqdm(users, total=users.count()):
        params = ParametersSchema(**user.parameters.__dict__)
        for record in user.records:
            user_feature_vector = session.query(UserFeatureVector).get(record.id)
            if user_feature_vector is None:
                print(user.id, record.id)
                continue
            user_feature_vector.parameters = json.dumps(params.__dict__)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
