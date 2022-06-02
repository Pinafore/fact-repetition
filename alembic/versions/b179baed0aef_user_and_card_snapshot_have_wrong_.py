"""user and card snapshot have wrong foreign key

Revision ID: b179baed0aef
Revises: 28f67ccad6cd
Create Date: 2022-06-02 21:16:31.393685

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'b179baed0aef'
down_revision = '28f67ccad6cd'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_constraint(u'usersnapshot_id_fkey', 'usersnapshot', type_='foreignkey')
    op.create_foreign_key(u'usersnapshot_id_fkey', 'usersnapshot', 'schedulerequest', ['id'], ['id'])
    op.drop_constraint(u'cardsnapshot_id_fkey', 'cardsnapshot', type_='foreignkey')
    op.create_foreign_key(u'cardsnapshot_id_fkey', 'cardsnapshot', 'schedulerequest', ['id'], ['id'])


def downgrade():
    pass
