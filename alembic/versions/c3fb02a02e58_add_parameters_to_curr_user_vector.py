"""add parameters to curr user vector

Revision ID: c3fb02a02e58
Revises: eb4b6af61710
Create Date: 2021-01-03 03:53:02.468582

"""
import json
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column
from sqlalchemy.dialects.postgresql import JSONB

from karl.models import User, Parameters, CurrUserFeatureVector
from karl.schemas import ParametersSchema

# revision identifiers, used by Alembic.
revision = 'c3fb02a02e58'
down_revision = 'eb4b6af61710'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.add_column(
        'curruserfeaturevector',
        Column('parameters', JSONB),
    )


def schema_downgrade():
    op.drop_column('curruserfeaturevector', 'parameters')


def data_upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session.query(User)
    for user in tqdm(users, total=users.count()):
        if user.parameters is None:
            user.parameters = Parameters(id=user.id, **ParametersSchema().__dict__)
        params = ParametersSchema(**user.parameters.__dict__)
        for record in user.records:
            user_feature_vector = session.query(CurrUserFeatureVector).get(user.id)
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
