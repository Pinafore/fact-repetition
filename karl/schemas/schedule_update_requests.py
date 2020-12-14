from pydantic import BaseModel
from typing import Optional


class ScheduleRequest(BaseModel):
    user_id: str
    fact_id: str
    date: str
    text: str
    answer: str
    category: str
    deck_id: Optional[str]
    deck_name: Optional[str]


class UpdateRequest(BaseModel):
    user_id: str
    fact_id: str
    label: bool
    debug_id: str
    history_id: str
    deck_id: Optional[str]
    deck_name: Optional[str]
    elapsed_milliseconds_text: int
    elapsed_milliseconds_answer: int
