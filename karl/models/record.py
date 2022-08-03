from sqlalchemy import Column, ForeignKey, String, Integer, Boolean, TIMESTAMP, ARRAY
from sqlalchemy.orm import relationship

from karl.db.base_class import Base
from karl.models import User, Card


class ScheduleRequest(Base):
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String)
    card_ids = Column(ARRAY(String), nullable=False)
    date = Column(TIMESTAMP(timezone=True))
    repetition_model = Column(String)

    study_records = relationship('StudyRecord', order_by='StudyRecord.date', back_populates='schedule_request')


class StudyRecord(Base):
    id = Column(String, primary_key=True, index=True)  # history_id / front_end_id provided by Matthew
    debug_id = Column(String, ForeignKey(ScheduleRequest.id, ondelete='CASCADE'), index=True)
    studyset_id = Column(String, index=True)  # session id
    user_id = Column(String, ForeignKey(User.id, ondelete='CASCADE'), index=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete='CASCADE'), index=True)
    deck_id = Column(String)
    label = Column(Boolean)
    date = Column(TIMESTAMP(timezone=True))
    elapsed_milliseconds_text = Column(Integer)
    elapsed_milliseconds_answer = Column(Integer)
    count = Column(Integer, nullable=False, default=0)
    count_session = Column(Integer, nullable=False, default=0)

    user = relationship("User", back_populates="study_records")
    card = relationship("Card", back_populates="study_records")
    schedule_request = relationship("ScheduleRequest", back_populates="study_records")


class TestRecord(Base):
    id = Column(String, primary_key=True, index=True)  # history_id / front_end_id provided by Matthew
    # debug_id = Column(String, ForeignKey(ScheduleRequest.id), index=True)  # test model doesn't create schedule requests, thus no schedule_request_id / debug_id
    studyset_id = Column(String, index=True)  # session id
    user_id = Column(String, ForeignKey(User.id, ondelete='CASCADE'), index=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete='CASCADE'), index=True)
    deck_id = Column(String)
    label = Column(Boolean)
    date = Column(TIMESTAMP(timezone=True))
    elapsed_milliseconds_text = Column(Integer)
    elapsed_milliseconds_answer = Column(Integer)
    count = Column(Integer, nullable=False, default=0)
    count_session = Column(Integer, nullable=False, default=0)

    user = relationship("User", back_populates="test_records")
    card = relationship("Card", back_populates="test_records")
