import json
import os

import yfinance as yf
from fastapi import APIRouter
from fastapi.params import Query

from .models import Data, Portfolio, History

router = APIRouter(prefix="/load")


@router.get("/data")
def get_data() -> list[Data]:
    tickers = [
        "NVDA", "AAPL", "MSFT", "AMZN", "GOOGL",
        "GOOG", "META", "AVGO", "TSLA",
        "LLY", "WMT", "JPM", "V", "ORCL",
        "MA", "XOM", "JNJ", "PLTR", "ABBV",
        "BAC", "NFLX", "COST", "AMD", "HD",
        "PG", "MU", "GE", "CSCO", "CVX"
    ]

    out: list[Data] = []
    miss_labelled = {
        'date': 'Date',
        'daily_min': 'Low',
        'daily_max': 'High'
    }
    for ticker in tickers:
        tick = yf.Ticker(ticker)
        history = tick.history(period="max")
        history.reset_index(inplace=True)
        if "Date" in history.columns:
            history.Date = history.Date.apply(lambda x: x.strftime('%Y-%m-%d'))

        for c in miss_labelled.keys():
            if c in history.columns:
                history.rename(columns={c: miss_labelled[c]})
        ################# Description #################
        desc = "No description provided"
        if "longBusinessSummary" in tick.info:
            desc = tick.info["longBusinessSummary"]
        ###############################################

        out.append(
            Data(
                name=ticker,
                description=desc,
                history=[History.model_validate(value) for value in
                         history[[
                             "Date",
                             "Open",
                             "Close",
                             "High",
                             "Low",
                             "Volume"
                         ]].to_dict(orient="records")]
            ))

    return out


@router.get("/portfolio")
def get_portfolio(
        portfolio_path: str = Query(
            default=...,
            description="It represent the path to the portfolio time series."
        )
) -> Portfolio:
    if os.path.exists(portfolio_path):
        with open(portfolio_path, "r") as f:
            data_json = json.load(f)
        return Portfolio.model_validate(data_json)
    else:
        raise ValueError("The path to the portfolio analysis does not exists")
