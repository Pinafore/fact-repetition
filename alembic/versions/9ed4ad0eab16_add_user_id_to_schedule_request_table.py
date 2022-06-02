"""add user_id to schedule request table

Revision ID: 9ed4ad0eab16
Revises: 4b4c9b5d72c9
Create Date: 2022-06-02 00:28:34.120020

"""
from alembic import op
import sqlalchemy as sa
from tqdm import tqdm

from karl.models import StudyRecord, ScheduleRequest
from karl.db.session import SessionLocal, engine


# revision identifiers, used by Alembic.
revision = '9ed4ad0eab16'
down_revision = '4b4c9b5d72c9'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('schedulerequest', sa.Column('user_id', sa.String()))

    # session = SessionLocal()
    # requests = session.query(ScheduleRequest)
    # for request in tqdm(requests, total=requests.count()):
    #     request.user_id = request.study_records[0].user_id
    # session.commit()


def downgrade():
    pass
