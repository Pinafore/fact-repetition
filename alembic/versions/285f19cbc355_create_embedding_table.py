"""create embedding table

Revision ID: 285f19cbc355
Revises: 8ee5c06f238a
Create Date: 2020-12-13 02:33:28.398267

"""
from tqdm import tqdm
from alembic import op
from sqlalchemy import orm
from sqlalchemy import Column, String
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from karl.old_models import Fact as OldCard
from karl.models import Embedding, BinaryNumpy
from karl.config import settings


# revision identifiers, used by Alembic.
revision = '285f19cbc355'
down_revision = '8ee5c06f238a'
branch_labels = None
depends_on = None


def schema_upgrade():
    op.create_table(
        'embedding',
        Column('id', String, primary_key=True, index=True),
        Column('embedding', BinaryNumpy, nullable=False)
    )


def schema_downgrade():
    op.drop_table('embedding')


def data_upgrade():
    engine = create_engine(settings.STABLE_DATABASE_URL)
    session_remote = sessionmaker(bind=engine, autoflush=False)()

    bind = op.get_bind()
    session = orm.Session(bind=bind)

    cards = session_remote.query(OldCard)
    for card_old in tqdm(cards, total=cards.count()):
        embedding = Embedding(
            id=card_old.fact_id,
            embedding=card_old.qrep,
        )
        session.add(embedding)

    session.commit()


def upgrade():
    schema_upgrade()
    data_upgrade()


def downgrade():
    schema_downgrade()
