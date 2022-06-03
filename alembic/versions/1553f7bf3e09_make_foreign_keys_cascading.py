"""make foreign keys cascading

Revision ID: 1553f7bf3e09
Revises: d9fa85b03d88
Create Date: 2022-06-03 19:08:21.074881

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '1553f7bf3e09'
down_revision = 'd9fa85b03d88'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint(u'embedding_id_fkey', 'embedding', type_='foreignkey')
    op.create_foreign_key(u'embedding_id_fkey', 'embedding', 'card', ['id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'leitner_user_id_fkey', 'leitner', type_='foreignkey')
    op.create_foreign_key(u'leitner_user_id_fkey', 'leitner', 'user', ['user_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'leitner_card_id_fkey', 'leitner', type_='foreignkey')
    op.create_foreign_key(u'leitner_card_id_fkey', 'leitner', 'card', ['card_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'sm2_user_id_fkey', 'sm2', type_='foreignkey')
    op.create_foreign_key(u'sm2_user_id_fkey', 'sm2', 'user', ['user_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'sm2_card_id_fkey', 'sm2', type_='foreignkey')
    op.create_foreign_key(u'sm2_card_id_fkey', 'sm2', 'card', ['card_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'parameters_id_fkey', 'parameters', type_='foreignkey')
    op.create_foreign_key(u'parameters_id_fkey', 'parameters', 'user', ['id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'studyrecord_debug_id_fkey', 'studyrecord', type_='foreignkey')
    op.create_foreign_key(u'studyrecord_debug_id_fkey', 'studyrecord', 'schedulerequest', ['debug_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'studyrecord_user_id_fkey', 'studyrecord', type_='foreignkey')
    op.create_foreign_key(u'studyrecord_user_id_fkey', 'studyrecord', 'user', ['user_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'studyrecord_card_id_fkey', 'studyrecord', type_='foreignkey')
    op.create_foreign_key(u'studyrecord_card_id_fkey', 'studyrecord', 'card', ['card_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'testrecord_user_id_fkey', 'testrecord', type_='foreignkey')
    op.create_foreign_key(u'testrecord_user_id_fkey', 'testrecord', 'user', ['user_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'testrecord_card_id_fkey', 'testrecord', type_='foreignkey')
    op.create_foreign_key(u'testrecord_card_id_fkey', 'testrecord', 'card', ['card_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'userstatsv2_user_id_fkey', 'userstatsv2', type_='foreignkey')
    op.create_foreign_key(u'userstatsv2_user_id_fkey', 'userstatsv2', 'user', ['user_id'], ['id'], ondelete="CASCADE")


def downgrade():
    pass
