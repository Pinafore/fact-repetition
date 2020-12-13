from sqlalchemy import Column, ForeignKey, String, Integer, TIMESTAMP

from karl.db.base_class import Base
from karl.models import User, Card


class Leitner(Base):
    user_id = Column(String, ForeignKey(User.id), primary_key=True, index=True)
    card_id = Column(String, ForeignKey(Card.id), primary_key=True, index=True)
    box = Column(Integer, nullable=False)
    scheduled_date = Column(TIMESTAMP(timezone=True))
