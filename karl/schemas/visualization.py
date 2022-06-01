from pydantic import BaseModel
from typing import Optional


class Visualization(BaseModel):
    name: str
    specs: str
    user_id: str
    deck_id: Optional[str] = None
    date_start: Optional[str] = None
    date_end: Optional[str] = None
