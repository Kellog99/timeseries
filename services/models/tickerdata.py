from datetime import datetime, date
from typing import Optional, List, Annotated, Union

from annotated_types import Ge
from pydantic import BaseModel, Field, ConfigDict


class History(BaseModel):
    model_config = ConfigDict(extra = 'ignore')

    Low: float = Field(description = 'Lowest trading price during the observation period.')
    High: float = Field(description = 'Highest trading price during the observation period.')
    Open: float = Field(description = 'Opening price of the observation period.')
    Close: float = Field(description = 'Closing price of the observation period.')
    Volume: int = Field(description = 'Trading volume during the observation period.')


class TickerData(BaseModel):
    name: str = Field(description = 'Trading symbol identifying the instrument.')
    description: str = Field(description = 'Human-readable instrument or company description.')
    history: dict[Union[str, date], History] = Field(
        description = 'Historical price and volume records indexed by date.'
    )


class TickerReturn(BaseModel):
    name: str = Field(description = 'Trading symbol of the instrument.')
    weeklyReturn: float = Field(description = 'Price return over the last week, expressed as a percentage.')
    monthlyReturn: float = Field(description = 'Price return over the last four weeks, expressed as a percentage.')
    yearlyReturn: float = Field(description = 'Price return over the last 365 days, expressed as a percentage.')
    totalReturn: float = Field(description = 'Price return over the available history, expressed as a percentage.')


class Holdings(BaseModel):
    stocks: Optional[float] = Field(
        default = None,
        ge = 0,
        description = "Value of stock holdings."
    )
    bonds: Optional[float] = Field(
        default = None,
        ge = 0,
        description = "Value of bond holdings."
    )
    currency: Optional[float] = Field(
        default = None,
        ge = 0,
        description = "Value of currency holdings."
    )
    crypto: Optional[float] = Field(
        default = None,
        ge = 0,
        description = "Value of cryptocurrency holdings."
    )
    etf: Optional[float] = Field(
        default = None,
        ge = 0,
        description = "Value of exchange-traded fund holdings."
    )


class Metrics(BaseModel):
    sharpe_ratio: Optional[float] = Field(None, description = 'Portfolio Sharpe ratio; None if not computed.')
    inv_return: Optional[float] = Field(None, description = 'Portfolio investment return; None if not computed.')
    variance: Optional[float] = Field(None, description = 'Variance of portfolio returns; None if not computed.')


class InvestmentHistory(BaseModel):
    Date: str = Field(description = 'Date of the portfolio history observation.')
    Invested: Annotated[float, Ge(0.)] = Field(description = 'Amount invested as of the observation date.')
    Value: Annotated[float, Ge(0.)] = Field(description = 'Market value of the portfolio on the observation date.')


class Portfolio(BaseModel):
    """Complete portfolio model"""
    portfolio_id: str = Field(
        default = ...,
        description = "Unique portfolio identifier"
    )
    last_updated: datetime = Field(
        default = ...,
        description = "Last update timestamp"
    )
    metrics: Metrics = Field(
        default = ...,
        description = "The metrics that have been computed on the portfolio."
    )
    holdings: Holdings = Field(
        default = ...,
        description = "The amount of money in the portfolio, for each category, value."
    )
    investment: Holdings = Field(
        default = ...,
        description = "The amount of money invested for each category."
    )
    history: List[InvestmentHistory] = Field(
        default = [],
        description = "Daily historical data"
    )
