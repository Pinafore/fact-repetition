from sqlalchemy import Column, ForeignKey, Integer, Float, Boolean, String, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB

from karl.db.base_class import Base
from karl.models import User, Card, Record


class UserCardFeatureVector(Base):
    id = Column(String, ForeignKey(Record.id, ondelete="CASCADE"), primary_key=True, index=True)
    user_id = Column(String, ForeignKey(User.id, ondelete="CASCADE"), index=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete="CASCADE"), index=True)
    date = Column(TIMESTAMP(timezone=True), index=True)
    n_study_positive = Column(Integer)
    n_study_negative = Column(Integer)
    n_study_total = Column(Integer)
    delta = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)
    leitner_box = Column(Integer)
    leitner_scheduled_date = Column(TIMESTAMP(timezone=True))
    sm2_efactor = Column(Float)
    sm2_interval = Column(Float)
    sm2_repetition = Column(Integer)
    sm2_scheduled_date = Column(TIMESTAMP(timezone=True))
    correct_on_first_try = Column(Boolean)


class UserFeatureVector(Base):
    id = Column(String, ForeignKey(Record.id, ondelete="CASCADE"), primary_key=True, index=True)
    user_id = Column(String, ForeignKey(User.id, ondelete="CASCADE"), index=True)
    date = Column(TIMESTAMP(timezone=True), index=True)
    n_study_positive = Column(Integer)
    n_study_negative = Column(Integer)
    n_study_total = Column(Integer)
    delta = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)
    parameters = Column(JSONB)


class CardFeatureVector(Base):
    id = Column(String, ForeignKey(Record.id, ondelete="CASCADE"), primary_key=True, index=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete="CASCADE"), index=True)
    date = Column(TIMESTAMP(timezone=True), index=True)
    n_study_positive = Column(Integer)
    n_study_negative = Column(Integer)
    n_study_total = Column(Integer)
    delta = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)


class CurrUserCardFeatureVector(Base):
    user_id = Column(String, ForeignKey(User.id, ondelete="CASCADE"), primary_key=True, index=True)
    card_id = Column(String, ForeignKey(Card.id, ondelete="CASCADE"), primary_key=True, index=True)
    n_study_positive = Column(Integer)
    n_study_negative = Column(Integer)
    n_study_total = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)
    correct_on_first_try = Column(Boolean)
    leitner_box = Column(Integer)
    leitner_scheduled_date = Column(TIMESTAMP(timezone=True))
    sm2_efactor = Column(Float)
    sm2_interval = Column(Float)
    sm2_repetition = Column(Integer)
    sm2_scheduled_date = Column(TIMESTAMP(timezone=True))


class CurrUserFeatureVector(Base):
    user_id = Column(String, ForeignKey(User.id, ondelete="CASCADE"), primary_key=True, index=True)
    n_study_positive = Column(Integer)
    n_study_negative = Column(Integer)
    n_study_total = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)
    parameters = Column(JSONB)


class CurrCardFeatureVector(Base):
    card_id = Column(String, ForeignKey(Card.id, ondelete="CASCADE"), primary_key=True, index=True)
    n_study_positive = Column(Integer)
    n_study_negative = Column(Integer)
    n_study_total = Column(Integer)
    previous_delta = Column(Integer)
    previous_study_date = Column(TIMESTAMP(timezone=True))
    previous_study_response = Column(Boolean)
