"""create sm2 table

Revision ID: f095364f3c69
Revises: 1a8279b0deda
Create Date: 2020-12-13 01:10:59.777713

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, String, Integer, Float, TIMESTAMP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import User as OldUser
from karl.models import User, Card, SM2

# revision identifiers, used by Alembic.
revision = 'f095364f3c69'
down_revision = '1a8279b0deda'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'sm2',
        Column('user_id', String, ForeignKey(User.id), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id), primary_key=True, index=True),
        Column('efactor', Float, nullable=False),
        Column('interval', Float, nullable=False),
        Column('repetition', Integer, nullable=False),
        Column('scheduled_date', TIMESTAMP(timezone=True)),
    )


def schema_downgrade():
    op.drop_table('sm2')


def data_upgrade():
    engine = create_engine('postgresql+psycopg2://shifeng@4.tcp.ngrok.io:18805/karl-prod')
    session_remote = sessionmaker(bind=engine, autoflush=False)()

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session_remote.query(OldUser)
    for user in tqdm(users, total=users.count()):
        for card_id, date in user.sm2_scheduled_date.items():
            if card_id not in user.sm2_efactor:
                continue
            if card_id not in user.sm2_interval:
                continue
            if card_id not in user.sm2_repetition:
                continue
            sm2 = SM2(
                user_id=user.user_id,
                card_id=card_id,
                efactor=user.sm2_efactor[card_id],
                interval=user.sm2_interval[card_id],
                repetition=user.sm2_repetition[card_id],
                scheduled_date=date,
            )
            session.add(sm2)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
