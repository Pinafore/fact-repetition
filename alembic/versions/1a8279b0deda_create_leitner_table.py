"""create leitner table

Revision ID: 1a8279b0deda
Revises: ea3ca513e7e1
Create Date: 2020-12-13 01:02:33.085247

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, ForeignKey, String, Integer, TIMESTAMP
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import User as OldUser
from karl.models import User, Card, Leitner
from karl.config import settings


# revision identifiers, used by Alembic.
revision = '1a8279b0deda'
down_revision = 'ea3ca513e7e1'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'leitner',
        Column('user_id', String, ForeignKey(User.id), primary_key=True, index=True),
        Column('card_id', String, ForeignKey(Card.id), primary_key=True, index=True),
        Column('box', Integer, nullable=False),
        Column('scheduled_date', TIMESTAMP(timezone=True)),
    )


def schema_downgrade():
    op.drop_table('leitner')


def data_upgrade():
    engine = create_engine(settings.STABLE_DATABASE_URL)
    session_remote = sessionmaker(bind=engine, autoflush=False)()

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    users = session_remote.query(OldUser)
    for user in tqdm(users, total=users.count()):
        if not user.user_id.isdigit():
            continue

        for card_id, box in user.leitner_box.items():
            if card_id in user.leitner_scheduled_date:
                leitner = Leitner(
                    user_id=user.user_id,
                    card_id=card_id,
                    box=box,
                    scheduled_date=user.leitner_scheduled_date[card_id],
                )
                session.add(leitner)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
