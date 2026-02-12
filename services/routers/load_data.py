import json
from pathlib import Path
from typing import Optional

import yfinance as yf
from fastapi import APIRouter, HTTPException, Depends
from fastapi.params import Query

from .models.data import Data, Portfolio, History
from .utils import validate_path, config_field

router = APIRouter(prefix="/load")


@router.get("/tickers")
def get_tickers(tickers: list[str] = Depends(config_field("tickers"))) -> list[str]:
    """
    This returns the ticker list that have been saved in the config file.
    """
    return tickers


def evaluate_portfolio() -> Portfolio:
    """
    This function evaluates the portfolio and returns it.
    """
    return Portfolio()


@router.get("/portfolio")
def get_portfolio_analysis(
        path_report: Path | str = Depends(config_field("path_report")),
) -> Portfolio:
    path_portfolio: Path = validate_path(path_report) / "analysis.json"
    if path_portfolio.exists():
        with open(path_portfolio, "r") as f:
            data_json = json.load(f)
        return Portfolio.model_validate(data_json)
    else:
        return evaluate_portfolio()


@router.get("/ticker_ts")
def get_data(
        ticker: str = Query(
            default=...,
            description="It represent the path to the ticker time series."
        ),
        path_data: Path | str = Depends(config_field("path_data")),
        step_size: Optional[int] = Depends(config_field("step_size")),
        save: bool = Depends(config_field("save"))
) -> Data:
    print("ticker = ", ticker)
    try:
        # Handle the existence of the path where the data should exist
        if isinstance(path_data, (str, Path)):
            if isinstance(path_data, str):
                path_data = Path(path_data).expanduser()
            path_data.mkdir(parents=True, exist_ok=True)
        else:
            raise ValueError(f"The type of path, {type(path_data)}, is not supported")

        ticker_path: Path = path_data / f"{ticker}.json"
        if ticker_path.exists():
            with open(ticker_path, "r") as f:
                data_json = json.load(f)
            out: Data = Data.model_validate(data_json)

        else:
            ##### Data download #####
            tick = yf.Ticker(ticker)
            history = tick.history(period="max")
            history.reset_index(inplace=True)
            if "Date" in history.columns:
                history.Date = history.Date.apply(
                    lambda x: x.strftime('%Y-%m-%d')
                )
            ##########################
            miss_labelled = {
                'date': 'Date',
                'daily_min': 'Low',
                'daily_max': 'High'
            }

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

            out = Data(
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
                with open(ticker_path, "w") as f:
                    json.dump(out.model_dump(), f)

        # Reducing the dimensionality of the history array
        out.history = [value for i, value in enumerate(out.history) if i % step_size == 0]
        return out

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
