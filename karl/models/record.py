from sqlalchemy import Column, ForeignKey, String, Integer, Boolean, TIMESTAMP, ARRAY
from sqlalchemy.orm import relationship

from karl.db.base_class import Base
from karl.models import User, Card


class Record(Base):
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey(User.id), index=True)
    card_id = Column(String, ForeignKey(Card.id), index=True)
    front_end_id = Column(String, index=True)
    deck_id = Column(String)
    response = Column(Boolean)
    elapsed_milliseconds_text = Column(Integer)
    elapsed_milliseconds_answer = Column(Integer)
    is_new_fact = Column(Boolean, nullable=False)
    date = Column(TIMESTAMP(timezone=True))

    user = relationship("User", back_populates="records")
    card = relationship("Card", back_populates="records")


class ScheduleRequest(Base):
    id = Column(String, primary_key=True, index=True)
    card_ids = Column(ARRAY(String), nullable=False)
    date = Column(TIMESTAMP(timezone=True))

    study_records = relationship('StudyRecord', order_by='StudyRecord.date', back_populates='schedule_request')


class StudyRecord(Base):
    id = Column(String, primary_key=True, index=True)  # history_id / front_end_id provided by Matthew
    debug_id = Column(String, ForeignKey(ScheduleRequest.id), index=True)
    studyset_id = Column(String, index=True)  # session id
    user_id = Column(String, ForeignKey(User.id), index=True)
    card_id = Column(String, ForeignKey(Card.id), index=True)
    deck_id = Column(String)
    label = Column(Boolean)
    date = Column(TIMESTAMP(timezone=True))
    elapsed_milliseconds_text = Column(Integer)
    elapsed_milliseconds_answer = Column(Integer)
    times_seen = Column(Integer, nullable=False, default=0)
    times_seen_in_session = Column(Integer, nullable=False, default=0)

    user = relationship("User", back_populates="study_records")
    card = relationship("Card", back_populates="study_records")
    schedule_request = relationship("ScheduleRequest", back_populates="study_records")


class TestRecord(Base):
    id = Column(String, primary_key=True, index=True)  # history_id / front_end_id provided by Matthew
    # debug_id = Column(String, ForeignKey(ScheduleRequest.id), index=True)  # test model doesn't create schedule requests, thus no schedule_request_id / debug_id
    studyset_id = Column(String, index=True)  # session id
    user_id = Column(String, ForeignKey(User.id), index=True)
    card_id = Column(String, ForeignKey(Card.id), index=True)
    deck_id = Column(String)
    label = Column(Boolean)
    date = Column(TIMESTAMP(timezone=True))
    elapsed_milliseconds_text = Column(Integer)
    elapsed_milliseconds_answer = Column(Integer)
    times_seen = Column(Integer, nullable=False, default=0)
    times_seen_in_session = Column(Integer, nullable=False, default=0)

    user = relationship("User", back_populates="test_records")
    card = relationship("Card", back_populates="test_records")
