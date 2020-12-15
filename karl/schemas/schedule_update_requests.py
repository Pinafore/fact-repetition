from pydantic import BaseModel
from typing import Optional


class ScheduleRequestSchema(BaseModel):
    text: str
    date: Optional[str]
    answer: str
    category: Optional[str]
    user_id: str
    fact_id: str
    # label: Optional[bool]
    # history_id: Optional[str]
    # repetition_model: Optional[str]
    deck_name: Optional[str]
    deck_id: Optional[str]
    # env: Optional[str] (moved out of the request)
    # elapsed_seconds_text: Optional[int]
    # elapsed_seconds_answer: Optional[int]
    elapsed_milliseconds_text: Optional[int]
    elapsed_milliseconds_answer: Optional[int]
    debug_id: Optional[str]
    history_id: Optional[str]
    label: Optional[bool]


class UpdateRequestSchema(BaseModel):
    text: str
    date: Optional[str]
    answer: Optional[str]
    category: Optional[str]
    user_id: str
    fact_id: str
    label: Optional[bool]
    history_id: Optional[str]
    # repetition_model: Optional[str]
    deck_name: Optional[str]
    deck_id: Optional[str]
    # env: Optional[str] (moved out of the request)
    # elapsed_seconds_text: Optional[int]
    # elapsed_seconds_answer: Optional[int]
    elapsed_milliseconds_text: int
    elapsed_milliseconds_answer: int
    debug_id: str
