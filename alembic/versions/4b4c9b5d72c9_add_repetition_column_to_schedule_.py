"""add repetition column to schedule request table

Revision ID: 4b4c9b5d72c9
Revises: b3defed892c7
Create Date: 2022-06-01 23:29:38.044527

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '4b4c9b5d72c9'
down_revision = 'b3defed892c7'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('schedulerequest', sa.Column('repetition_model', sa.String()))


def downgrade():
    pass
