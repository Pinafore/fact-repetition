"""change times_seen to count in record

Revision ID: 7a83ccb1065a
Revises: e10fe3d5355e
Create Date: 2022-06-02 17:09:39.282635

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7a83ccb1065a'
down_revision = 'e10fe3d5355e'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column('studyrecord', 'times_seen', new_column_name='count')
    op.alter_column('studyrecord', 'times_seen_in_session', new_column_name='count_session')
    op.alter_column('testrecord', 'times_seen', new_column_name='count')
    op.alter_column('testrecord', 'times_seen_in_session', new_column_name='count_session')


def downgrade():
    pass
