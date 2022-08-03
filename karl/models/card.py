from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from karl.db.base_class import Base


class Card(Base):
    id = Column(String, primary_key=True, index=True)
    text = Column(String, nullable=False)
    answer = Column(String, nullable=False)
    category = Column(String)
    deck_name = Column(String)
    deck_id = Column(String)

    study_records = relationship('StudyRecord', order_by='StudyRecord.date', back_populates='card')
    test_records = relationship('TestRecord', order_by='TestRecord.date', back_populates='card')
