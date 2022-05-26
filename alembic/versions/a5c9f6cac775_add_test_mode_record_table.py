"""add test mode record table

Revision ID: a5c9f6cac775
Revises: 
Create Date: 2022-05-25 18:00:18.625414

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a5c9f6cac775'
down_revision = None
branch_labels = None
depends_on = None


class Record(Base):
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey(User.id), index=True)
    card_id = Column(String, ForeignKey(Card.id), index=True)
    front_end_id = Column(String, index=True)
    deck_id = Column(String)
    response = Column(Boolean)
    elapsed_milliseconds_text = Column(Integer)
    elapsed_milliseconds_answer = Column(Integer)
    is_new_fact = Column(Boolean, nullable=False)
    date = Column(TIMESTAMP(timezone=True))

    user = relationship("User", back_populates="records")
    card = relationship("Card", back_populates="records")


def upgrade():
    def schema_upgrades():
    op.create_table(
        'user_snapshots',
        sa.Column('debug_id', sa.String, primary_key=True),
        sa.Column('record_id', sa.String, sa.ForeignKey('records.record_id')),
        sa.Column('user_id', sa.String, sa.ForeignKey('users.user_id')),
        sa.Column('date', sa.DateTime),
        sa.Column('recent_facts', MutableList.as_mutable(JSONEncoded)),
        sa.Column('previous_study', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('leitner_box', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('leitner_scheduled_date', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_efactor', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_interval', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_repetition', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('sm2_scheduled_date', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('results', MutableList.as_mutable(JSONEncoded)),
        sa.Column('count_correct_before', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('count_wrong_before', MutableDict.as_mutable(JSONEncoded)),
        sa.Column('params', ParamsType),
    )


def downgrade():
    op.drop_table('user_snapshots')
