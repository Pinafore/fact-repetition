from pydantic import BaseModel
from typing import List, Optional

class KarlFactSchema(BaseModel):
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


class ScheduleRequestSchema(BaseModel):
    user_id: str
    facts: List[KarlFactSchema]  # this list can be empty
    repetition_model: str
    recall_target: RecallTarget


class UpdateRequestSchema(BaseModel):
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
    fact: Optional[KarlFactSchema]
