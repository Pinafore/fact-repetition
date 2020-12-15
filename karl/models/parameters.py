from sqlalchemy import Column, Integer, Float, String, ForeignKey
from sqlalchemy.orm import relationship

from karl.db.base_class import Base
from karl.models import User


class Parameters(Base):
    id = Column(String, ForeignKey(User.id), primary_key=True, index=True)
    repetition_model = Column(String, nullable=False)
    card_embedding = Column(Float, default=1, nullable=False)
    recall = Column(Float, default=1, nullable=False)
    recall_target = Column(Float, default=0.85, nullable=False)
    category = Column(Float, default=1, nullable=False)
    answer = Column(Float, default=1, nullable=False)
    leitner = Column(Float, default=0, nullable=False)
    sm2 = Column(Float, default=0, nullable=False)
    decay_qrep = Column(Float, default=0.9, nullable=False)
    cool_down = Column(Float, default=1, nullable=False)
    cool_down_time_correct = Column(Integer, default=20, nullable=False)
    cool_down_time_wrong = Column(Integer, default=1, nullable=False)
    max_recent_facts = Column(Integer, default=10, nullable=False)

    user = relationship('User', back_populates='parameters')
