"""fsrs

Revision ID: ce6ca5b3a293
Revises: d3759fe12bce
Create Date: 2023-11-01 16:40:15.025387

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'ce6ca5b3a293'
down_revision = 'd3759fe12bce'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('usercardfeaturevector', sa.Column('fsrs_scheduled_date', postgresql.TIMESTAMP(timezone=True), nullable=True))
    op.add_column('usercardfeaturevector', sa.Column('stability', sa.FLOAT(), nullable=True))
    op.add_column('usercardfeaturevector', sa.Column('difficulty', sa.FLOAT(), nullable=True))
    op.add_column('usercardfeaturevector', sa.Column('state', sa.INTEGER(), nullable=True))


def downgrade() -> None:
    op.drop_column('usercardfeaturevector', 'fsrs_scheduled_date')
    op.drop_column('usercardfeaturevector', 'stability')
    op.drop_column('usercardfeaturevector', 'difficulty')
    op.drop_column('usercardfeaturevector', 'state')
