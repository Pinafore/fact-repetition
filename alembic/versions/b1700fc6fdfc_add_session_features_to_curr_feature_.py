"""add session features to curr feature vector

Revision ID: b1700fc6fdfc
Revises: 7a83ccb1065a
Create Date: 2022-06-02 17:38:56.319386

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b1700fc6fdfc'
down_revision = '7a83ccb1065a'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('currusercardfeaturevector', sa.Column('count_positive_session', sa.Integer()))
    op.add_column('currusercardfeaturevector', sa.Column('count_negative_session', sa.Integer()))
    op.add_column('currusercardfeaturevector', sa.Column('count_session', sa.Integer()))
    op.add_column('currusercardfeaturevector', sa.Column('previous_delta_session', sa.Integer()))
    op.add_column('currusercardfeaturevector', sa.Column('previous_study_date_session', sa.TIMESTAMP(timezone=True)))
    op.add_column('currusercardfeaturevector', sa.Column('previous_study_response_session', sa.Boolean()))
    op.add_column('curruserfeaturevector', sa.Column('count_positive_session', sa.Integer()))
    op.add_column('curruserfeaturevector', sa.Column('count_negative_session', sa.Integer()))
    op.add_column('curruserfeaturevector', sa.Column('count_session', sa.Integer()))
    op.add_column('curruserfeaturevector', sa.Column('previous_delta_session', sa.Integer()))
    op.add_column('curruserfeaturevector', sa.Column('previous_study_date_session', sa.TIMESTAMP(timezone=True)))
    op.add_column('curruserfeaturevector', sa.Column('previous_study_response_session', sa.Boolean()))


def downgrade():
    pass
