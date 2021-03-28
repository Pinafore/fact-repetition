from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class VUserCard(BaseModel):
    user_id: str
    card_id: str
    n_study_positive: int
    n_study_negative: int
    n_study_total: int
    previous_delta: Optional[int]
    previous_study_date: Optional[datetime]
    previous_study_response: Optional[bool]
    correct_on_first_try: Optional[bool]
    leitner_box: Optional[int]
    leitner_scheduled_date: Optional[datetime]
    sm2_efactor: Optional[float]
    sm2_interval: Optional[float]
    sm2_repetition: Optional[int]
    sm2_scheduled_date: Optional[datetime]


class VUser(BaseModel):
    user_id: str
    n_study_positive: int
    n_study_negative: int
    n_study_total: int
    previous_delta: Optional[int]
    previous_study_date: Optional[datetime]
    previous_study_response: Optional[bool]
    parameters: str


class VCard(BaseModel):
    card_id: str
    n_study_positive: int
    n_study_negative: int
    n_study_total: int
    previous_delta: Optional[int]
    previous_study_date: Optional[datetime]
    previous_study_response: Optional[bool]
