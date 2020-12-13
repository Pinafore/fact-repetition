from sqlalchemy import Column, Integer, Float, String, ForeignKey

from karl.db.base_class import Base
from karl.models import User, Card


class FeatureVector(Base):
    user_id = Column(String, ForeignKey(User.id), primary_key=True, index=True)
    card_id = Column(String, ForeignKey(Card.id), primary_key=True, index=True)
