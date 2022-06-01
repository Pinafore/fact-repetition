from sqlalchemy import Column, ForeignKey, ARRAY, String, TIMESTAMP

from karl.db.base_class import Base


class ScheduleRequest(Base):
    id = Column(String, primary_key=True, index=True)
    card_ids = Column(ARRAY(String), nullable=False)
    date = Column(TIMESTAMP(timezone=True))
