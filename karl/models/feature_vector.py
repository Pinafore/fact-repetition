from sqlalchemy import Column, ForeignKey, Integer, Float, Boolean, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from karl.db.base_class import Base
from karl.models import User, Card, ScheduleRequest


class UserCardFeatureVector(Base):
    user_id = Column(String, ForeignKey(User.id, ondelete='CASCADE'), primary_key=True, index=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete='CASCADE'), primary_key=True, index=True)
    count_positive = Column(Integer, default=0)
    count_negative = Column(Integer, default=0)
    count = Column(Integer, default=0)
    count_positive_session = Column(Integer, default=0)
    count_negative_session = Column(Integer, default=0)
    count_session = Column(Integer, default=0)
    # no delta here since it need to be computed when schedule request arrives
    previous_delta = Column(Integer, default=None)
    previous_study_date = Column(TIMESTAMP(timezone=True), default=None)
    previous_study_response = Column(Boolean, default=None)
    previous_delta_session = Column(Integer, default=None)
    previous_study_date_session = Column(TIMESTAMP(timezone=True), default=None)
    previous_study_response_session = Column(Boolean, default=None)
    correct_on_first_try = Column(Boolean, default=None)
    correct_on_first_try_session = Column(Boolean, default=None)
    leitner_box = Column(Integer, default=None)
    leitner_scheduled_date = Column(TIMESTAMP(timezone=True), default=None)
    sm2_efactor = Column(Float, default=None)
    sm2_interval = Column(Float, default=None)
    sm2_repetition = Column(Integer, default=None)
    sm2_scheduled_date = Column(TIMESTAMP(timezone=True), default=None)
    schedule_request_id = Column(String, default=None)


class UserFeatureVector(Base):
    user_id = Column(String, ForeignKey(User.id, ondelete='CASCADE'), primary_key=True, index=True)
    count_positive = Column(Integer, default=0)
    count_negative = Column(Integer, default=0)
    count = Column(Integer, default=0)
    count_positive_session = Column(Integer, default=0)
    count_negative_session = Column(Integer, default=0)
    count_session = Column(Integer, default=0)
    previous_delta = Column(Integer, default=None)
    previous_study_date = Column(TIMESTAMP(timezone=True), default=None)
    previous_study_response = Column(Boolean, default=None)
    previous_delta_session = Column(Integer, default=None)
    previous_study_date_session = Column(TIMESTAMP(timezone=True), default=None)
    previous_study_response_session = Column(Boolean, default=None)
    parameters = Column(JSONB)
    schedule_request_id = Column(String, default=None)


class CardFeatureVector(Base):
    card_id = Column(String, ForeignKey(Card.id, ondelete='CASCADE'), primary_key=True, index=True)
    count_positive = Column(Integer, default=0)
    count_negative = Column(Integer, default=0)
    count = Column(Integer, default=0)
    previous_delta = Column(Integer, default=None)
    previous_study_date = Column(TIMESTAMP(timezone=True), default=None)
    previous_study_response = Column(Boolean, default=None)


class UserCardSnapshotV2(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey(User.id, ondelete='CASCADE'), index=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete='CASCADE'), index=True)
    schedule_request_id = Column(String, ForeignKey(ScheduleRequest.id, ondelete='CASCADE'), index=True)
    date = Column(TIMESTAMP(timezone=True), index=True)
    count_positive = Column(Integer)
    count_negative = Column(Integer)
    count = Column(Integer)
    count_positive_session = Column(Integer)
    count_negative_session = Column(Integer)
    count_session = Column(Integer)
    delta = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)
    delta_session = Column(Integer)
    previous_delta_session = Column(Integer)
    previous_study_date_session = Column(TIMESTAMP(timezone=True))
    previous_study_response_session = Column(Boolean)
    leitner_box = Column(Integer)
    leitner_scheduled_date = Column(TIMESTAMP(timezone=True))
    sm2_efactor = Column(Float)
    sm2_interval = Column(Float)
    sm2_repetition = Column(Integer)
    sm2_scheduled_date = Column(TIMESTAMP(timezone=True))
    correct_on_first_try = Column(Boolean)
    correct_on_first_try_session = Column(Boolean)

    schedule_request = relationship('ScheduleRequest', back_populates='usercard_snapshots')


class UserSnapshotV2(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey(User.id, ondelete='CASCADE'), index=True)
    schedule_request_id = Column(String, ForeignKey(ScheduleRequest.id, ondelete='CASCADE'), index=True)
    date = Column(TIMESTAMP(timezone=True), index=True)
    count_positive = Column(Integer)
    count_negative = Column(Integer)
    count = Column(Integer)
    count_positive_session = Column(Integer)
    count_negative_session = Column(Integer)
    count_session = Column(Integer)
    delta = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)
    delta_session = Column(Integer)
    previous_delta_session = Column(Integer)
    previous_study_date_session = Column(TIMESTAMP(timezone=True))
    previous_study_response_session = Column(Boolean)
    parameters = Column(JSONB)

    schedule_request = relationship('ScheduleRequest', back_populates='user_snapshots')



class CardSnapshotV2(Base):
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete='CASCADE'), index=True)
    schedule_request_id = Column(String, ForeignKey(ScheduleRequest.id, ondelete='CASCADE'), index=True)
    date = Column(TIMESTAMP(timezone=True), index=True)
    count_positive = Column(Integer)
    count_negative = Column(Integer)
    count = Column(Integer)
    delta = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)

    schedule_request = relationship('ScheduleRequest', back_populates='card_snapshots')
