import json
import os

import yfinance as yf
from fastapi import APIRouter, HTTPException
from fastapi.params import Query

from .models import Data, Portfolio, History

router = APIRouter(prefix="/load")


@router.get("/data")
def get_data(
        ticker: str = Query(
            default=...,
            description="Ticker to ask for the data"
        ),
        data_path: str = Query(
            default="/home/andrea/Desktop/projects/timeseries/ts_storage/data",
            description="Time series already saved"
        ),
        save: bool = Query(
            default=True,
            description="It tells whether a new ticker has to be saved into the data path folder."
        )) -> Data:
    try:
        data_path = os.path.join(data_path, f"{ticker}.json")
        if os.path.exists(data_path):
            with open(data_path, "r") as f:
                data_json = json.load(f)
            return Data.model_validate(data_json)

        miss_labelled = {
            'date': 'Date',
            'daily_min': 'Low',
            'daily_max': 'High'
        }
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
        try:
            info = tick.info
            if "longBusinessSummary" in info:
                desc = tick.info["longBusinessSummary"]

        except Exception as e:
            # Handle yfinance-specific errors
            raise HTTPException(
                status_code=503,
                detail=f"Unable to fetch data for {ticker}: {str(e)}"
            )
        ###############################################

        data = Data(
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
        )
        if save:
            with open(data_path, "w") as f:
                json.dump(data.model_dump(), f)

        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
