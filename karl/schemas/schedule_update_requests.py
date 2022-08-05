from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Optional

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


class RepetitionModel(str, Enum):
    leitner = "leitner"
    karl = "karl"
    sm2 = "sm-2"
    karl100 = "karl100"
    karl50 = "karl50"
    karl85 = "karl85"
    settles = "settles"

    @classmethod
    def select_model(cls):
        return choice([Repetition.leitner, Repetition.karl, Repetition.settles])

class ScheduleRequestSchema(BaseModel):
    user_id: str
    facts: List[KarlFactSchema]  # this list can be empty
    repetition_model: RepetitionModel
    recall_target: RecallTarget


class ScheduleResponseSchema(BaseModel):
    debug_id: str  # same as schedule_request_id
    order: List[int]
    scores: List[float]
    details: Optional[List]
    rationale: Optional[str]
    profile: Optional[dict]


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
    typed: Optional[str]  # user-entered answer
    recommendation: Optional[bool]  # system's judgment of whether the user-entered answer is correct
