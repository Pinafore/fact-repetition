from sqlalchemy import Column, String
from sqlalchemy.orm import relationship

from karl.db.base_class import Base


class User(Base):
    id = Column(String, primary_key=True, index=True)

    records = relationship('Record', order_by='Record.date', back_populates='user')
    parameters = relationship('Parameters', back_populates='user', uselist=False)
