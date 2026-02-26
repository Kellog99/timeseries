from typing import Optional

from fastapi import APIRouter
from fastapi.params import Query

from .models.indicators import Fundamentals

router = APIRouter(prefix="/indicators")


@router.get("/")
def get_indicators(
        ticker: str = Query(
            ...,
            description="Ticker's id",
        ),
        data: Optional[list[float]] = None,
) -> Fundamentals:
    """
    This function has the role to compute all the fundamentals of a certain time series which can be defined in two ways:
     1) a certain ticker or timeseries
     2) data
    """
    if data:
        return Fundamentals(
            market_cap=2850000000000,
            pe_ratio=29.45,
            pb_ratio=45.23,
            dividend_yield=0.52,
            eps=6.19,
            beta=1.29,
            week_52_high=199.62,
            week_52_low=164.08,
            avg_volume=58420000,
            revenue_growth=8.12,
            profit_margin=25.31,
            roe=147.25,
            debt_to_equity=1.81,
        )
    else:
        print(ticker)
        return Fundamentals(
            market_cap=2850000000000,
            pe_ratio=29.45,
            pb_ratio=45.23,
            dividend_yield=0.52,
            eps=6.19,
            beta=1.29,
            week_52_high=199.62,
            week_52_low=164.08,
            avg_volume=58420000,
            revenue_growth=8.12,
            profit_margin=25.31,
            roe=147.25,
            debt_to_equity=1.81,
        )
