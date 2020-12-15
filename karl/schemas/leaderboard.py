from typing import Optional, List
from pydantic import BaseModel


class IntOrFloat:

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if isinstance(v, float) or isinstance(v, int):
            return v
        raise TypeError('int or float required')


class RankingSchema(BaseModel):
    user_id: int
    rank: int
    # value: Union[int, float]  # don't use this
    value: IntOrFloat


class LeaderboardSchema(BaseModel):
    leaderboard: List[RankingSchema]
    total: int
    rank_type: str
    user_place: Optional[int] = None
    user_id: Optional[str] = None
    skip: Optional[int] = 0
    limit: Optional[int] = None
