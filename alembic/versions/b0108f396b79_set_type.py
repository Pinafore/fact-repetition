"""set_type

Revision ID: b0108f396b79
Revises: ce6ca5b3a293
Create Date: 2023-11-18 02:55:01.288846

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b0108f396b79'
down_revision = 'ce6ca5b3a293'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('schedulerequest', sa.Column('test_mode', sa.INTEGER(), nullable=True))
    op.add_column('schedulerequest', sa.Column('set_type', sa.VARCHAR(), nullable=True))
    op.add_column('testrecord', sa.Column('set_type', sa.VARCHAR()))


def downgrade() -> None:
    op.drop_column('schedulerequest', 'test_mode')
    op.drop_column('schedulerequest', 'set_type')
    op.drop_column('test_record', 'set_type')
