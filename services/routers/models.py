from pydantic import BaseModel


class History(BaseModel):
    daily_min: float
    daily_max: float
    daily_avg: float


class Data(BaseModel):
    name: str
    description: str
    history: dict[str, History]
