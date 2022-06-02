"""add current session id to curr feature vector

Revision ID: e10fe3d5355e
Revises: 39faba6bc6c7
Create Date: 2022-06-02 16:53:42.295114

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e10fe3d5355e'
down_revision = '39faba6bc6c7'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('currusercardfeaturevector', sa.Column('studyset_id', sa.Integer()))
    op.add_column('curruserfeaturevector', sa.Column('studyset_id', sa.Integer()))
    op.add_column('currusercardfeaturevector', sa.Column('correct_on_first_try_session', sa.Boolean()))


def downgrade():
    pass
