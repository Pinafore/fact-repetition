from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from karl.db.base_class import Base


class User(Base):
    id = Column(String, primary_key=True, index=True)

    study_records = relationship('StudyRecord', order_by='StudyRecord.date', back_populates='user')
    test_records = relationship('TestRecord', order_by='TestRecord.date', back_populates='user')
    parameters = relationship('Parameters', back_populates='user', uselist=False)
