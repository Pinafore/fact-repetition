from sqlalchemy import Column, ForeignKey, ARRAY, String, TIMESTAMP

from karl.db.base_class import Base
from karl.models import Record


class ScheduleRequest(Base):
    id = Column(String, ForeignKey(Record.id, ondelete='CASADE'), primary_key=True, index=True)
    card_ids = Column(ARRAY(String), nullable=False)
    date = Column(TIMESTAMP(timezone=True))
