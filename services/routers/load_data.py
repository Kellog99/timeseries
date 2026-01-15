import json
from pathlib import Path
from typing import Optional

import yfinance as yf
from fastapi import APIRouter, HTTPException, Request
from fastapi.params import Query

from .models.data import Data, Portfolio, History
from .models.main_config import MainConfig

router = APIRouter(prefix="/load")


@router.get("/tickers")
def get_tickers(request: Request) -> list[str]:
    return request.app.state.config.tickers


@router.get("/data")
def get_data_service(
        request: Request,
        ticker: str = Query(
            default=...,
            description="It represent the path to the ticker time series."
        ),
        step_size: Optional[int] = None
):
    config: MainConfig = request.app.state.config
    return get_data(
        ticker=ticker,
        path_data=Path(config.path_data).expanduser(),
        step_size=step_size if step_size is not None else config.step_lineChart,
        save=config.save
    )


@router.get("/portfolio")
def get_portfolio(request: Request) -> Portfolio:
    config: MainConfig = request.app.state.config
    path_portfolio: Path = Path(config.path_portfolio).expanduser()
    if path_portfolio.exists():
        with open(path_portfolio, "r") as f:
            data_json = json.load(f)
        return Portfolio.model_validate(data_json)
    else:
        raise ValueError("The path to the portfolio analysis does not exists")


def get_data(
        ticker: str,
        path_data: Optional[Path] = Path("~/Desktop/Data").expanduser(),
        step_size: Optional[int] = None,
        save: bool = True,
        **kwargs
) -> Data:
    try:
        if path_data:
            if isinstance(path_data, str):
                path_data = Path(path_data).expanduser()

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
