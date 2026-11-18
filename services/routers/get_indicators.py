import math

import yfinance as yf
from fastapi import APIRouter, HTTPException

from services.models.indicators import Fundamentals

router = APIRouter(
    prefix = '/indicators'
)


@router.get(
    '/'
)
def get_indicators(
    ticker: str
) -> Fundamentals:
    """Provider fundamentals. Missing values are null; rates and ratios are decimals."""
    try:
        info = yf.Ticker(
            ticker
        ).info
    except Exception as exc:
        raise HTTPException(
            503,
            'Unable to retrieve fundamentals.'
        ) from exc
    fields = {
        'market_cap': 'marketCap', 'pe_ratio': 'trailingPE', 'pb_ratio': 'priceToBook',
        'dividend_yield': 'trailingAnnualDividendYield', 'eps': 'trailingEps', 'beta': 'beta',
        'week_52_high': 'fiftyTwoWeekHigh', 'week_52_low': 'fiftyTwoWeekLow',
        'avg_volume': 'averageVolume', 'revenue_growth': 'revenueGrowth',
        'profit_margin': 'profitMargins', 'roe': 'returnOnEquity', 'debt_to_equity': 'debtToEquity',
    }
    values = {}
    for field, key in fields.items():
        value = info.get(
            key
        )
        values[field] = float(
            value
        ) if isinstance(
            value,
            (int, float)
        ) and math.isfinite(
            value
        ) else None
    if values['debt_to_equity'] is not None:
        values['debt_to_equity'] /= 100  # Yahoo quotes this ratio in percentage units.
    if all(
            value is None for value in values.values()
    ):
        raise HTTPException(
            404,
            'No fundamentals available for this ticker.'
        )
    return Fundamentals(
        **values
    )
