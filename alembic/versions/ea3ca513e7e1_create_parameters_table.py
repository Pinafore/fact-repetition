"""create parameters table

Revision ID: ea3ca513e7e1
Revises: 2d454f153c1a
Create Date: 2020-12-13 00:52:21.831075

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, String, Float, Integer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import User as OldUser
from karl.models import Parameters, User
from karl.config import settings


# revision identifiers, used by Alembic.
revision = 'ea3ca513e7e1'
down_revision = '2d454f153c1a'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'parameters',
        Column('id', String, ForeignKey(User.id), primary_key=True, index=True),
        Column('repetition_model', String, nullable=False),
        Column('card_embedding', Float, default=1, nullable=False),
        Column('recall', Float, default=1, nullable=False),
        Column('recall_target', Float, default=0.85, nullable=False),
        Column('category', Float, default=1, nullable=False),
        Column('answer', Float, default=1, nullable=False),
        Column('leitner', Float, default=0, nullable=False),
        Column('sm2', Float, default=0, nullable=False),
        Column('decay_qrep', Float, default=0.9, nullable=False),
        Column('cool_down', Float, default=1, nullable=False),
        Column('cool_down_time_correct', Integer, default=20, nullable=False),
        Column('cool_down_time_wrong', Integer, default=1, nullable=False),
        Column('max_recent_facts', Integer, default=10, nullable=False),
    )


def schema_downgrade():
    op.drop_table('parameters')


def data_upgrade():
    engine = create_engine(settings.STABLE_DATABASE_URL)
    session_remote = sessionmaker(bind=engine, autoflush=False)()

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session_remote.query(OldUser)
    for user_old in tqdm(users, total=users.count()):
        if not user_old.user_id.isdigit():
            continue

        params_old = user_old.params.__dict__
        params_new = Parameters(
            id=user_old.user_id,
            repetition_model=params_old.get('repetition_model', 'karl85'),
            card_embedding=params_old.get('qreq', 1.0),
            recall=params_old.get('recall', 1.0),
            recall_target=params_old.get('recall_target', 0.85),
            category=params_old.get('category', 1.0),
            answer=params_old.get('answer', 1.0),
            leitner=params_old.get('leitner', 0.0),
            sm2=params_old.get('sm2', 0.0),
            decay_qrep=params_old.get('decay_qrep', 0.9),
            cool_down=params_old.get('cool_down', 1.0),
            cool_down_time_correct=params_old.get('cool_down_time_correct', 20),
            cool_down_time_wrong=params_old.get('cool_down_time_wrong', 1),
            max_recent_facts=params_old.get('max_recent_facts', 10),
        )
        session.add(params_new)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
