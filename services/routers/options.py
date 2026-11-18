"""Yahoo snapshots for research; retrieval time is not an exchange quote timestamp."""
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
import yfinance as yf
from fastapi import APIRouter, HTTPException
from pydantic import AwareDatetime, ValidationError

from timeseries.instruments import OptionContract
from timeseries.options import OptionMarketQuote

router = APIRouter(
    prefix = '/load'
)


@router.get(
    '/options'
)
def load_option_chain(
    ticker: str,
    expiry: AwareDatetime,
    exercise: Literal['european', 'american'] = 'american',
    settlement: Literal['cash', 'physical'] = 'physical',
    currency: str = 'USD'
) -> list[OptionMarketQuote]:
    """Caller supplies exact expiry; invalid/crossed/empty quotes are omitted."""
    try:
        chain = yf.Ticker(
            ticker
        ).option_chain(
            expiry.date().isoformat()
        )
    except Exception as exc:
        raise HTTPException(
            503,
            'Unable to retrieve the option chain.'
        ) from exc
    observed = datetime.now(
        UTC
    )
    result = []
    frames: list[tuple[Literal['call', 'put'], pd.DataFrame]] = [('call', chain.calls), ('put', chain.puts)]
    for option_type, frame in frames:
        for row in frame.to_dict(
                orient = 'records'
        ):
            try:
                contract = OptionContract(
                    symbol = row['contractSymbol'],
                    underlying = ticker,
                    strike = row['strike'],
                    expiry = expiry,
                    option_type = option_type,
                    exercise = exercise,
                    settlement = settlement,
                    currency = currency
                )
                result.append(
                    OptionMarketQuote(
                        contract = contract,
                        observed_at = observed,
                        bid = row['bid'],
                        ask = row['ask'],
                        volume = row.get(
                            'volume'
                        ) if pd.notna(
                            row.get(
                                'volume'
                            )
                        ) else None,
                        open_interest = row.get(
                            'openInterest'
                        ) if pd.notna(
                            row.get(
                                'openInterest'
                            )
                        ) else None,
                        last_trade_at = row.get(
                            'lastTradeDate'
                        ) if pd.notna(
                            row.get(
                                'lastTradeDate'
                            )
                        ) else None,
                    )
                )
            except (ValidationError, KeyError):
                continue
    return result
