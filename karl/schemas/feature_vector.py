from typing import Optional
from datetime import datetime
from pydantic import BaseModel

class VUserCard(BaseModel):
    user_id: str
    card_id: str
    count_positive: int
    count_negative: int
    count: int
    count_positive_session: int
    count_negative_session: int
    count_session: int
    delta: Optional[int]
    previous_delta: Optional[int]
    previous_study_date: Optional[datetime]
    previous_study_response: Optional[bool]
    correct_on_first_try: Optional[bool]
    previous_delta_session: Optional[int]
    previous_study_date_session: Optional[datetime]
    previous_study_response_session: Optional[bool]
    correct_on_first_try_session: Optional[bool]
    leitner_box: Optional[int]
    leitner_scheduled_date: Optional[datetime]
    sm2_efactor: Optional[float]
    sm2_interval: Optional[float]
    sm2_repetition: Optional[int]
    sm2_scheduled_date: Optional[datetime]
    date: Optional[datetime]
    schedule_request_id: Optional[str]


class VUser(BaseModel):
    user_id: str
    count_positive: int
    count_negative: int
    count: int
    count_positive_session: int
    count_negative_session: int
    count_session: int
    previous_delta: Optional[int]
    previous_study_date: Optional[datetime]
    previous_study_response: Optional[bool]
    previous_delta_session: Optional[int]
    previous_study_date_session: Optional[datetime]
    previous_study_response_session: Optional[bool]
    parameters: str
    date: Optional[datetime]
    schedule_request_id: Optional[str]


class VCard(BaseModel):
    card_id: str
    count_positive: int
    count_negative: int
    count: int
    previous_delta: Optional[int]
    previous_study_date: Optional[datetime]
    previous_study_response: Optional[bool]
    date: Optional[datetime]
