from sqlalchemy import Column, ForeignKey, String, Integer, Boolean, TIMESTAMP
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
