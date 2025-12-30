from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class History(BaseModel):
    Date: str
    Low: float
    High: float
    Open: float
    Close: float
    Volume: int


class Data(BaseModel):
    name: str
    description: str
    history: list[History]


class Holdings(BaseModel):
    stocks: Optional[float] = Field(
        default=None,
        ge=0,
        description="Value of stocks"
    )
    bonds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Value of stocks"
    )
    currency: Optional[float] = Field(
        default=None,
        ge=0,
        description="Value of stocks"
    )
    crypto: Optional[float] = Field(
        default=None,
        ge=0,
        description="Value of stocks"
    )
    etf: Optional[float] = Field(
        default=None,
        ge=0,
        description="Value of stocks"
    )


class Metrics(BaseModel):
    sharpe_ratio: Optional[float] = None
    inv_return: Optional[float] = None
    variance: Optional[float] = None


class Portfolio(BaseModel):
    """Complete portfolio model"""
    portfolio_id: str = Field(
        default=...,
        description="Unique portfolio identifier"
    )
    last_updated: datetime = Field(
        default=...,
        description="Last update timestamp"
    )
    metrics: Metrics = Field(
        default=...,
        description="The metrics that have been computed on the portfolio."
    )
    holdings: Holdings = Field(
        default=...,
        description="The amount of money for each type of investment in the portfolio."
    )
    history: List[History] = Field(
        default=[],
        description="Daily historical data"
    )
