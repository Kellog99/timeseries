from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field


class History(BaseModel):
    daily_min: float
    daily_max: float
    daily_avg: float


class Data(BaseModel):
    name: str
    description: str
    history: dict[str, History]


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
    cripto: Optional[float] = Field(
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

    class Config:
        json_schema_extra = {
            "example": {
                "portfolio_id": "PORT_001",
                "last_updated": "2025-12-30T00:00:00Z",
                "metrics": {
                    "sharpe_ratio": 0.9,
                    "inv_return": 0.02,
                    "variance": 0.33
                },
                "current_holdings": {
                    "stocks": 15000.50,
                    "bonds": 8000.75,
                    "currency": 3000.25,
                    "crypto": 2000.00,
                    "etf": 12000.99,
                    "total_value": 40002.49
                },
                "history": [
                    {
                        "date": "2025-12-28",
                        "mean": 39500.25,
                        "max": 39850.60,
                        "min": 39200.10
                    }
                ]
            }
        }
