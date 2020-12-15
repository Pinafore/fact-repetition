from pydantic import BaseModel
from typing import List, Optional, Dict


class ScheduleResponseSchema(BaseModel):
    debug_id: str
    order: List[int]
    scores: List[Dict[str, float]]
    details: Optional[List]
    rationale: Optional[str]
