"""add n days studied to userstat

Revision ID: 8f756504c67f
Revises: 
Create Date: 2020-10-19 16:15:16.334355

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy import orm

from tqdm import tqdm
from karl.models import User, UserStat

# revision identifiers, used by Alembic.
revision = '8f756504c67f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    bind = op.get_bind()
    session = orm.Session(bind=bind)

    op.add_column('user_stats', sa.Column('n_days_studied', sa.Integer))

    for deck_id in session.query(UserStat.deck_id).distinct():
        print('deck_id', deck_id)
        users = session.query(User)
        for user in tqdm(users, total=users.count()):
            n_days_studied = 0
            prev_total_seen = 0

            for user_stat in session.query(UserStat).\
                    filter(UserStat.user_id == user.user_id).\
                    filter(UserStat.deck_id == deck_id).\
                    order_by(UserStat.date):
                if user_stat.total_seen - prev_total_seen > 0:
                    n_days_studied += 1
                user_stat.n_days_studied = n_days_studied
                prev_total_seen = user_stat.total_seen

    session.commit()


def downgrade():
    op.drop_column('user_stats', 'n_days_studied')
