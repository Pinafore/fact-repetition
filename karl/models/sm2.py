from sqlalchemy import Column, ForeignKey, String, Integer, Float, TIMESTAMP

from karl.db.base_class import Base
from karl.models import User, Card


class SM2(Base):
    user_id = Column(String, ForeignKey(User.id), primary_key=True, index=True)
    card_id = Column(String, ForeignKey(Card.id), primary_key=True, index=True)
    efactor = Column(Float, nullable=False)
    interval = Column(Float, nullable=False)
    repetition = Column(Integer, nullable=False)
    scheduled_date = Column(TIMESTAMP(timezone=True))
