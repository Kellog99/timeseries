import json
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import yfinance as yf
from fastapi import APIRouter, HTTPException, Depends
from fastapi.params import Query
from pandas import DataFrame

from .models.tickerdata import TickerData, Portfolio, History
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
) -> TickerData:
    """
    This function create the structure associated with the Ticker's information
    Args:
        ticker: The ticker that has to be load
        path_data: path to the cached file
        step_size: step size for the history
        save: flag that tells whether the data has to be saved

    Return:
        The data in the `Data` format
    """

    ############## Path Data ##############
    # this part has a twofold role:
    # 1) handle the typing of the variable
    # 2) handle the existence of the folder
    if not path_data:
        path_data: Path = Path("./data")

    if isinstance(path_data, (str, Path)):
        if isinstance(path_data, str):
            path_data = Path(path_data).expanduser()
        path_data.mkdir(parents=True, exist_ok=True)
    else:
        raise ValueError(f"The type of path, {type(path_data)}, is not supported")

    #######################################

    ############## Load Cached Ticker ##############
    today_date: date = date.today()
    out: TickerData | None = None

    ticker_path: Path = path_data / f"{ticker}.json"
    if ticker_path.exists() and ticker_path.is_file():
        try:
            # Trying to parse the json file
            with open(ticker_path, "r", encoding="utf-8") as f:
                data_json = json.load(f)

        except json.JSONDecodeError as e:
            print(f"\nJSON Error: {e}")
            print(f"Position: {e.pos}")
            print(f"Line: {e.lineno}, Column: {e.colno}")
        out = TickerData.model_validate(data_json)

    ################################################

    last_date: date = today_date
    if out:
        # Creating a list of date
        dates: list[date] = [
            datetime.strptime(x, '%Y-%m-%d').date() if isinstance(x, str) else x
            for x in out.history.keys()
        ]
        last_date = max(dates)

    # Downloading the data in two cases:
    # 1) there is no cached ticker
    # 2) if the cached ticker is not updated
    if not out or today_date > last_date:
        ##### Data download #####
        tick = yf.Ticker(ticker)
        if not out:
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
            out: TickerData = TickerData(
                name=ticker,
                description=desc,
                history={}
            )

        if len(out.history) == 0:
            history: DataFrame = tick.history(period="max")
        else:
            history: DataFrame = tick.history(start=last_date, end=today_date)

        history.reset_index(inplace=True)
        if "Date" in history.columns:
            history.Date = history.Date.apply(
                lambda x: x.strftime('%Y-%m-%d')
            )

        ########################## Miss Labelled columns ##########################
        miss_labelled = {
            'date': 'Date',
            'daily_min': 'Low',
            'daily_max': 'High'
        }
        for c in miss_labelled.keys():
            if c in history.columns:
                history.rename(columns={c: miss_labelled[c]})

        ###########################################################################

        # These are the information that will be stored as value
        required_columns: list[str] = ["Open", "Close", "High", "Low", "Volume"]

        new_history: dict[str | date, History] = {
            value.get("Date"): History.model_validate(value) for value in
            history[required_columns + ["Date"]].to_dict(orient="records")
        }

        # adding the missing history
        out.history = out.history | new_history

        if save:
            with open(ticker_path, "w", encoding="utf-8") as f:
                json.dump(
                    obj={
                        "name": out.name,
                        "description": out.description,
                        "history": {
                            k.isoformat() if isinstance(k, date) else k: v.model_dump()
                            for k, v in out.history.items()
                        }
                    },
                    fp=f
                )

    # Reducing the dimensionality of the history array
    keys = sorted(
        out.history.keys(),
        key=lambda x: datetime.strptime(x, "%Y-%m-%d").date()
    )

    out.history = {
        key: out.history[key] for i, key in enumerate(keys) if i % step_size == 0
    }
    return out
