from pydantic import BaseModel
from typing import List, Optional

class KarlFactV2(BaseModel):
    fact_id: str
    text: str
    answer: str
    deck_name: str
    deck_id: int
    category: Optional[str] = None


class RecallTarget(BaseModel):
    target: float
    target_window_lowest: float
    target_window_highest: float


class ScheduleRequestV2(BaseModel):
    user_id: str
    facts: List[KarlFactV2]  # this list can be empty
    repetition_model: str
    recall_target: RecallTarget


class UpdateRequestV2(BaseModel):
    user_id: str
    fact_id: str
    deck_name: str
    deck_id: int
    label: bool
    elapsed_milliseconds_text: int
    elapsed_milliseconds_answer: int
    history_id: str  # uniquely identifies a study
    studyset_id: str
    debug_id: Optional[str] # aka schedule_request_id, n/a in test updates
    test_mode: bool
    fact: KarlFactV2


class ScheduleRequestSchema(BaseModel):
    text: str
    date: Optional[str]
    answer: str
    category: Optional[str]
    user_id: str
    fact_id: str
    deck_name: Optional[str]
    deck_id: Optional[str]
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
    # elapsed_seconds_text: Optional[int]
    # elapsed_seconds_answer: Optional[int]
    elapsed_milliseconds_text: int
    elapsed_milliseconds_answer: int
    debug_id: str
