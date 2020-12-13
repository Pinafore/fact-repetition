from sqlalchemy import Column, Integer, String, ForeignKey, Date

from karl.db.base_class import Base
from karl.models import User, Record


class UserStats(Base):
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey(User.id), index=True)
    deck_id = Column(String, nullable=False, index=True)
    date = Column(Date, nullable=False)
    n_cards_total = Column(Integer, nullable=False, default=0)
    n_cards_positive = Column(Integer, nullable=False, default=0)
    n_new_cards_total = Column(Integer, nullable=False, default=0)
    n_old_cards_total = Column(Integer, nullable=False, default=0)
    n_new_cards_positive = Column(Integer, nullable=False, default=0)
    n_old_cards_positive = Column(Integer, nullable=False, default=0)
    elapsed_milliseconds_text = Column(Integer, nullable=False, default=0)
    elapsed_milliseconds_answer = Column(Integer, nullable=False, default=0)
    n_days_studied = Column(Integer, nullable=False, default=0)
