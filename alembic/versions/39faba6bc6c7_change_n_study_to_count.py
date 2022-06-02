"""change n_study to count

Revision ID: 39faba6bc6c7
Revises: 9ed4ad0eab16
Create Date: 2022-06-02 16:46:05.259687

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '39faba6bc6c7'
down_revision = '9ed4ad0eab16'
branch_labels = None
depends_on = None


def upgrade():
    op.alter_column('usercardfeaturevector', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('usercardfeaturevector', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('usercardfeaturevector', 'n_study_total', new_column_name='count')

    op.alter_column('userfeaturevector', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('userfeaturevector', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('userfeaturevector', 'n_study_total', new_column_name='count')

    op.alter_column('cardfeaturevector', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('cardfeaturevector', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('cardfeaturevector', 'n_study_total', new_column_name='count')

    op.alter_column('currusercardfeaturevector', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('currusercardfeaturevector', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('currusercardfeaturevector', 'n_study_total', new_column_name='count')

    op.alter_column('curruserfeaturevector', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('curruserfeaturevector', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('curruserfeaturevector', 'n_study_total', new_column_name='count')

    op.alter_column('currcardfeaturevector', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('currcardfeaturevector', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('currcardfeaturevector', 'n_study_total', new_column_name='count')

    op.alter_column('usercardsnapshot', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('usercardsnapshot', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('usercardsnapshot', 'n_study_total', new_column_name='count')
    op.alter_column('usercardsnapshot', 'n_study_positive_session', new_column_name='count_positive_session')
    op.alter_column('usercardsnapshot', 'n_study_negative_session', new_column_name='count_negative_session')
    op.alter_column('usercardsnapshot', 'n_study_total_session', new_column_name='count_session')

    op.alter_column('usersnapshot', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('usersnapshot', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('usersnapshot', 'n_study_total', new_column_name='count')
    op.alter_column('usersnapshot', 'n_study_positive_session', new_column_name='count_positive_session')
    op.alter_column('usersnapshot', 'n_study_negative_session', new_column_name='count_negative_session')
    op.alter_column('usersnapshot', 'n_study_total_session', new_column_name='count_session')

    op.alter_column('cardsnapshot', 'n_study_positive', new_column_name='count_positive')
    op.alter_column('cardsnapshot', 'n_study_negative', new_column_name='count_negative')
    op.alter_column('cardsnapshot', 'n_study_total', new_column_name='count')

def downgrade():
    pass
