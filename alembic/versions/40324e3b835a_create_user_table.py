"""create user table

Revision ID: 40324e3b835a
Revises: 
Create Date: 2020-12-13 00:28:32.634993

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, String, Float, Integer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import User as OldUser
from karl.models import User, Parameters


# revision identifiers, used by Alembic.
revision = '40324e3b835a'
down_revision = None
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'user',
        Column('id', String, primary_key=True, index=True),
    )


def schema_downgrade():
    op.drop_table('user')


def data_upgrade():
    engine = create_engine('postgresql+psycopg2://shifeng@4.tcp.ngrok.io:18805/karl-prod')
    session_remote = sessionmaker(bind=engine, autoflush=False)()

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session_remote.query(OldUser)
    for user_old in tqdm(users, total=users.count()):
        user_new = User(id=user_old.user_id)
        session.add(user_new)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
