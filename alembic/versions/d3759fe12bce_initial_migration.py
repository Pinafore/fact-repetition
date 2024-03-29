"""initial migration

Revision ID: d3759fe12bce
Revises: 
Create Date: 2023-09-30 21:21:57.157530

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'd3759fe12bce'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('user',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.PrimaryKeyConstraint('id', name='user_pkey')
    )
    op.create_index('ix_user_id', 'user', ['id'], unique=False)

    op.create_table('card',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('text', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('answer', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('category', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('deck_name', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('deck_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='card_pkey')
    )
    op.create_index('ix_card_id', 'card', ['id'], unique=False)

    op.create_table('schedulerequest',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('card_ids', postgresql.ARRAY(sa.VARCHAR()), autoincrement=False, nullable=False),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('repetition_model', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('recall_target', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('recall_target_lowest', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('recall_target_highest', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.PrimaryKeyConstraint('id', name='schedulerequest_pkey')
    )
    op.create_index('ix_schedulerequest_id', 'schedulerequest', ['id'], unique=False)

    op.create_table('studyrecord',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('debug_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('studyset_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('deck_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('label', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('elapsed_milliseconds_text', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('elapsed_milliseconds_answer', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('typed', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('recommendation', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='studyrecord_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['debug_id'], ['schedulerequest.id'], name='studyrecord_debug_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='studyrecord_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='studyrecord_pkey')
    )
    op.create_index('ix_studyrecord_user_id', 'studyrecord', ['user_id'], unique=False)
    op.create_index('ix_studyrecord_studyset_id', 'studyrecord', ['studyset_id'], unique=False)
    op.create_index('ix_studyrecord_id', 'studyrecord', ['id'], unique=False)
    op.create_index('ix_studyrecord_debug_id', 'studyrecord', ['debug_id'], unique=False)
    op.create_index('ix_studyrecord_card_id', 'studyrecord', ['card_id'], unique=False)

    op.create_table('usersnapshotv2',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('schedule_request_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_positive_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date_session', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['schedule_request_id'], ['schedulerequest.id'], name='usersnapshotv2_schedule_request_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='usersnapshotv2_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='usersnapshotv2_pkey')
    )
    op.create_index('ix_usersnapshotv2_user_id', 'usersnapshotv2', ['user_id'], unique=False)
    op.create_index('ix_usersnapshotv2_schedule_request_id', 'usersnapshotv2', ['schedule_request_id'], unique=False)
    op.create_index('ix_usersnapshotv2_id', 'usersnapshotv2', ['id'], unique=False)
    op.create_index('ix_usersnapshotv2_date', 'usersnapshotv2', ['date'], unique=False)

    op.create_table('usercardsnapshotv2',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('schedule_request_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_positive_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date_session', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('leitner_box', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('leitner_scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('sm2_efactor', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('sm2_interval', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('sm2_repetition', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('sm2_scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('correct_on_first_try', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('correct_on_first_try_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='usercardsnapshotv2_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['schedule_request_id'], ['schedulerequest.id'], name='usercardsnapshotv2_schedule_request_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='usercardsnapshotv2_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='usercardsnapshotv2_pkey')
    )
    op.create_index('ix_usercardsnapshotv2_user_id', 'usercardsnapshotv2', ['user_id'], unique=False)
    op.create_index('ix_usercardsnapshotv2_schedule_request_id', 'usercardsnapshotv2', ['schedule_request_id'], unique=False)
    op.create_index('ix_usercardsnapshotv2_id', 'usercardsnapshotv2', ['id'], unique=False)
    op.create_index('ix_usercardsnapshotv2_date', 'usercardsnapshotv2', ['date'], unique=False)
    op.create_index('ix_usercardsnapshotv2_card_id', 'usercardsnapshotv2', ['card_id'], unique=False)

    op.create_table('leitner',
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('box', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='leitner_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='leitner_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id', 'card_id', name='leitner_pkey')
    )
    op.create_index('ix_leitner_user_id', 'leitner', ['user_id'], unique=False)
    op.create_index('ix_leitner_card_id', 'leitner', ['card_id'], unique=False)

    op.create_table('cardsnapshot',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='cardsnapshot_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['id'], ['schedulerequest.id'], name='cardsnapshot_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='cardsnapshot_pkey')
    )
    op.create_index('ix_cardsnapshot_id', 'cardsnapshot', ['id'], unique=False)
    op.create_index('ix_cardsnapshot_date', 'cardsnapshot', ['date'], unique=False)
    op.create_index('ix_cardsnapshot_card_id', 'cardsnapshot', ['card_id'], unique=False)

    op.create_table('usercardsnapshot',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_positive_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date_session', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('leitner_box', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('leitner_scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('sm2_efactor', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('sm2_interval', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('sm2_repetition', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('sm2_scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('correct_on_first_try', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('correct_on_first_try_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='usercardsnapshot_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['id'], ['schedulerequest.id'], name='usercardsnapshot_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='usercardsnapshot_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='usercardsnapshot_pkey')
    )
    op.create_index('ix_usercardsnapshot_user_id', 'usercardsnapshot', ['user_id'], unique=False)
    op.create_index('ix_usercardsnapshot_id', 'usercardsnapshot', ['id'], unique=False)
    op.create_index('ix_usercardsnapshot_date', 'usercardsnapshot', ['date'], unique=False)
    op.create_index('ix_usercardsnapshot_card_id', 'usercardsnapshot', ['card_id'], unique=False)

    op.create_table('cardsnapshotv2',
    sa.Column('id', sa.INTEGER(), autoincrement=True, nullable=False),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('schedule_request_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='cardsnapshotv2_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['schedule_request_id'], ['schedulerequest.id'], name='cardsnapshotv2_schedule_request_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='cardsnapshotv2_pkey')
    )
    op.create_index('ix_cardsnapshotv2_schedule_request_id', 'cardsnapshotv2', ['schedule_request_id'], unique=False)
    op.create_index('ix_cardsnapshotv2_id', 'cardsnapshotv2', ['id'], unique=False)
    op.create_index('ix_cardsnapshotv2_date', 'cardsnapshotv2', ['date'], unique=False)
    op.create_index('ix_cardsnapshotv2_card_id', 'cardsnapshotv2', ['card_id'], unique=False)

    op.create_table('embedding',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('embedding', postgresql.BYTEA(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['id'], ['card.id'], name='embedding_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='embedding_pkey')
    )
    op.create_index('ix_embedding_id', 'embedding', ['id'], unique=False)

    op.create_table('sm2',
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('efactor', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('interval', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('repetition', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='sm2_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='sm2_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id', 'card_id', name='sm2_pkey')
    )
    op.create_index('ix_sm2_user_id', 'sm2', ['user_id'], unique=False)
    op.create_index('ix_sm2_card_id', 'sm2', ['card_id'], unique=False)

    op.create_table('usercardfeaturevector',
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('correct_on_first_try', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('leitner_box', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('leitner_scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('sm2_efactor', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('sm2_interval', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=True),
    sa.Column('sm2_repetition', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('sm2_scheduled_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('schedule_request_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('correct_on_first_try_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('count_positive_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date_session', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='currusercardfeaturevector_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='currusercardfeaturevector_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id', 'card_id', name='currusercardfeaturevector_pkey')
    )
    op.create_index('ix_usercardfeaturevector_user_id', 'usercardfeaturevector', ['user_id'], unique=False)
    op.create_index('ix_usercardfeaturevector_card_id', 'usercardfeaturevector', ['card_id'], unique=False)

    op.create_table('usersnapshot',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_positive_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date_session', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['id'], ['schedulerequest.id'], name='usersnapshot_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='usersnapshot_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='usersnapshot_pkey')
    )
    op.create_index('ix_usersnapshot_user_id', 'usersnapshot', ['user_id'], unique=False)
    op.create_index('ix_usersnapshot_id', 'usersnapshot', ['id'], unique=False)
    op.create_index('ix_usersnapshot_date', 'usersnapshot', ['date'], unique=False)

    op.create_table('userstatsv2',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('deck_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('date', sa.DATE(), autoincrement=False, nullable=False),
    sa.Column('n_cards_total', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('n_cards_positive', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('n_new_cards_total', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('n_old_cards_total', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('n_new_cards_positive', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('n_old_cards_positive', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('elapsed_milliseconds_text', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('elapsed_milliseconds_answer', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('n_days_studied', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='userstatsv2_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='userstatsv2_pkey')
    )
    op.create_index('ix_userstatsv2_user_id', 'userstatsv2', ['user_id'], unique=False)
    op.create_index('ix_userstatsv2_id', 'userstatsv2', ['id'], unique=False)
    op.create_index('ix_userstatsv2_deck_id', 'userstatsv2', ['deck_id'], unique=False)

    op.create_table('parameters',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('repetition_model', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('card_embedding', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('recall', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('recall_target', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('category', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('answer', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('leitner', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('sm2', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('decay_qrep', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('cool_down', postgresql.DOUBLE_PRECISION(precision=53), autoincrement=False, nullable=False),
    sa.Column('cool_down_time_correct', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('cool_down_time_wrong', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('max_recent_facts', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['id'], ['user.id'], name='parameters_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='parameters_pkey')
    )
    op.create_index('ix_parameters_id', 'parameters', ['id'], unique=False)

    op.create_table('testrecord',
    sa.Column('id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('studyset_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('deck_id', sa.VARCHAR(), autoincrement=False, nullable=True),
    sa.Column('label', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('elapsed_milliseconds_text', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('elapsed_milliseconds_answer', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=False),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='testrecord_card_id_fkey', ondelete='CASCADE'),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='testrecord_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('id', name='testrecord_pkey')
    )
    op.create_index('ix_testrecord_user_id', 'testrecord', ['user_id'], unique=False)
    op.create_index('ix_testrecord_studyset_id', 'testrecord', ['studyset_id'], unique=False)
    op.create_index('ix_testrecord_id', 'testrecord', ['id'], unique=False)
    op.create_index('ix_testrecord_card_id', 'testrecord', ['card_id'], unique=False)

    op.create_table('userfeaturevector',
    sa.Column('user_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.Column('parameters', postgresql.JSONB(astext_type=sa.Text()), autoincrement=False, nullable=True),
    sa.Column('schedule_request_id', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_positive_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta_session', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date_session', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response_session', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['user_id'], ['user.id'], name='curruserfeaturevector_user_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('user_id', name='curruserfeaturevector_pkey')
    )
    op.create_index('ix_userfeaturevector_user_id', 'userfeaturevector', ['user_id'], unique=False)

    op.create_table('cardfeaturevector',
    sa.Column('card_id', sa.VARCHAR(), autoincrement=False, nullable=False),
    sa.Column('count_positive', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count_negative', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('count', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_delta', sa.INTEGER(), autoincrement=False, nullable=True),
    sa.Column('previous_study_date', postgresql.TIMESTAMP(timezone=True), autoincrement=False, nullable=True),
    sa.Column('previous_study_response', sa.BOOLEAN(), autoincrement=False, nullable=True),
    sa.ForeignKeyConstraint(['card_id'], ['card.id'], name='currcardfeaturevector_card_id_fkey', ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('card_id', name='currcardfeaturevector_pkey')
    )
    op.create_index('ix_cardfeaturevector_card_id', 'cardfeaturevector', ['card_id'], unique=False)

    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index('ix_card_id', table_name='card')
    op.drop_table('card')
    op.drop_index('ix_schedulerequest_id', table_name='schedulerequest')
    op.drop_table('schedulerequest')
    op.drop_index('ix_user_id', table_name='user')
    op.drop_table('user')
    op.drop_index('ix_cardfeaturevector_card_id', table_name='cardfeaturevector')
    op.drop_table('cardfeaturevector')
    op.drop_index('ix_userfeaturevector_user_id', table_name='userfeaturevector')
    op.drop_table('userfeaturevector')
    op.drop_index('ix_studyrecord_card_id', table_name='studyrecord')
    op.drop_index('ix_studyrecord_debug_id', table_name='studyrecord')
    op.drop_index('ix_studyrecord_id', table_name='studyrecord')
    op.drop_index('ix_studyrecord_studyset_id', table_name='studyrecord')
    op.drop_index('ix_studyrecord_user_id', table_name='studyrecord')
    op.drop_table('studyrecord')
    op.drop_index('ix_testrecord_card_id', table_name='testrecord')
    op.drop_index('ix_testrecord_id', table_name='testrecord')
    op.drop_index('ix_testrecord_studyset_id', table_name='testrecord')
    op.drop_index('ix_testrecord_user_id', table_name='testrecord')
    op.drop_table('testrecord')
    op.drop_index('ix_parameters_id', table_name='parameters')
    op.drop_table('parameters')
    op.drop_index('ix_userstatsv2_deck_id', table_name='userstatsv2')
    op.drop_index('ix_userstatsv2_id', table_name='userstatsv2')
    op.drop_index('ix_userstatsv2_user_id', table_name='userstatsv2')
    op.drop_table('userstatsv2')
    op.drop_index('ix_usersnapshot_date', table_name='usersnapshot')
    op.drop_index('ix_usersnapshot_id', table_name='usersnapshot')
    op.drop_index('ix_usersnapshot_user_id', table_name='usersnapshot')
    op.drop_table('usersnapshot')
    op.drop_index('ix_usercardfeaturevector_card_id', table_name='usercardfeaturevector')
    op.drop_index('ix_usercardfeaturevector_user_id', table_name='usercardfeaturevector')
    op.drop_table('usercardfeaturevector')
    op.drop_index('ix_sm2_card_id', table_name='sm2')
    op.drop_index('ix_sm2_user_id', table_name='sm2')
    op.drop_table('sm2')
    op.drop_index('ix_embedding_id', table_name='embedding')
    op.drop_table('embedding')
    op.drop_index('ix_cardsnapshotv2_card_id', table_name='cardsnapshotv2')
    op.drop_index('ix_cardsnapshotv2_date', table_name='cardsnapshotv2')
    op.drop_index('ix_cardsnapshotv2_id', table_name='cardsnapshotv2')
    op.drop_index('ix_cardsnapshotv2_schedule_request_id', table_name='cardsnapshotv2')
    op.drop_table('cardsnapshotv2')
    op.drop_index('ix_usercardsnapshot_card_id', table_name='usercardsnapshot')
    op.drop_index('ix_usercardsnapshot_date', table_name='usercardsnapshot')
    op.drop_index('ix_usercardsnapshot_id', table_name='usercardsnapshot')
    op.drop_index('ix_usercardsnapshot_user_id', table_name='usercardsnapshot')
    op.drop_table('usercardsnapshot')
    op.drop_index('ix_cardsnapshot_card_id', table_name='cardsnapshot')
    op.drop_index('ix_cardsnapshot_date', table_name='cardsnapshot')
    op.drop_index('ix_cardsnapshot_id', table_name='cardsnapshot')
    op.drop_table('cardsnapshot')
    op.drop_index('ix_leitner_card_id', table_name='leitner')
    op.drop_index('ix_leitner_user_id', table_name='leitner')
    op.drop_table('leitner')
    op.drop_index('ix_usercardsnapshotv2_card_id', table_name='usercardsnapshotv2')
    op.drop_index('ix_usercardsnapshotv2_date', table_name='usercardsnapshotv2')
    op.drop_index('ix_usercardsnapshotv2_id', table_name='usercardsnapshotv2')
    op.drop_index('ix_usercardsnapshotv2_schedule_request_id', table_name='usercardsnapshotv2')
    op.drop_index('ix_usercardsnapshotv2_user_id', table_name='usercardsnapshotv2')
    op.drop_table('usercardsnapshotv2')
    op.drop_index('ix_usersnapshotv2_date', table_name='usersnapshotv2')
    op.drop_index('ix_usersnapshotv2_id', table_name='usersnapshotv2')
    op.drop_index('ix_usersnapshotv2_schedule_request_id', table_name='usersnapshotv2')
    op.drop_index('ix_usersnapshotv2_user_id', table_name='usersnapshotv2')
    op.drop_table('usersnapshotv2')
    # ### end Alembic commands ###
