"""create card table

Revision ID: 2d454f153c1a
Revises: 40324e3b835a
Create Date: 2020-12-13 00:35:26.375564

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, String
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import Fact as OldCard
from karl.models import Card
from karl.config import settings


# revision identifiers, used by Alembic.
revision = '2d454f153c1a'
down_revision = '40324e3b835a'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'card',
        Column('id', String, primary_key=True, index=True),
        Column('text', String, nullable=False),
        Column('answer', String, nullable=False),
        Column('category', String),
        Column('deck_name', String),
        Column('deck_id', String),
    )


def schema_downgrade():
    op.drop_table('card')


def data_upgrade():
    engine = create_engine(settings.STABLE_DATABASE_URL)
    session_remote = sessionmaker(bind=engine, autoflush=False)()

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    cards = session_remote.query(OldCard)
    for card_old in tqdm(cards, total=cards.count()):
        card_old = card_old.__dict__
        card_new = Card(
            id=card_old['fact_id'],
            text=card_old['text'],
            answer=card_old['answer'],
            category=card_old.get('category', None),
            deck_name=card_old.get('deck_name', None),
            deck_id=card_old.get('deck_id', None),
        )
        session.add(card_new)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
