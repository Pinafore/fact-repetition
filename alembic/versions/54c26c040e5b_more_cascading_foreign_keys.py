"""more cascading foreign keys

Revision ID: 54c26c040e5b
Revises: 1553f7bf3e09
Create Date: 2022-06-03 19:30:08.638212

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '54c26c040e5b'
down_revision = '1553f7bf3e09'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint(u'currusercardfeaturevector_user_id_fkey', 'currusercardfeaturevector', type_='foreignkey')
    op.create_foreign_key(u'currusercardfeaturevector_user_id_fkey', 'currusercardfeaturevector', 'user', ['user_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'currusercardfeaturevector_card_id_fkey', 'currusercardfeaturevector', type_='foreignkey')
    op.create_foreign_key(u'currusercardfeaturevector_card_id_fkey', 'currusercardfeaturevector', 'card', ['card_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'curruserfeaturevector_user_id_fkey', 'curruserfeaturevector', type_='foreignkey')
    op.create_foreign_key(u'curruserfeaturevector_user_id_fkey', 'curruserfeaturevector', 'user', ['user_id'], ['id'], ondelete="CASCADE")
    op.drop_constraint(u'currcardfeaturevector_card_id_fkey', 'currcardfeaturevector', type_='foreignkey')
    op.create_foreign_key(u'currcardfeaturevector_card_id_fkey', 'currcardfeaturevector', 'card', ['card_id'], ['id'], ondelete="CASCADE")


def downgrade():
    pass
