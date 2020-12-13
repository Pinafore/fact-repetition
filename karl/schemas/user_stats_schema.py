from pydantic import BaseModel
from typing import Optional


class UserStatsSchema(BaseModel):
    user_id: str
    deck_id: str
    date_start: str
    date_end: str
    new_facts: Optional[int] = 0
    reviewed_facts: Optional[int] = 0
    new_correct: Optional[int] = 0
    reviewed_correct: Optional[int] = 0
    total_seen: Optional[int] = 0
    total_milliseconds: Optional[int] = 0
    total_seconds: Optional[int] = 0
    total_minutes: Optional[int] = 0
    elapsed_milliseconds_text: Optional[int] = 0
    elapsed_milliseconds_answer: Optional[int] = 0
    elapsed_seconds_text: Optional[int] = 0
    elapsed_seconds_answer: Optional[int] = 0
    elapsed_minutes_text: Optional[int] = 0
    elapsed_minutes_answer: Optional[int] = 0
    n_days_studied: Optional[int] = 0
    known_rate: Optional[float] = 0
    new_known_rate: Optional[float] = 0
    review_known_rate: Optional[float] = 0
