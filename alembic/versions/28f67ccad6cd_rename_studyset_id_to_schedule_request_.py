"""rename studyset_id to schedule_request_id in feature vector

Revision ID: 28f67ccad6cd
Revises: b1700fc6fdfc
Create Date: 2022-06-02 20:08:03.713529

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '28f67ccad6cd'
down_revision = 'b1700fc6fdfc'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column('currusercardfeaturevector', 'studyset_id', new_column_name='schedule_request_id')
    op.alter_column('curruserfeaturevector', 'studyset_id', new_column_name='schedule_request_id')


def downgrade():
    pass
