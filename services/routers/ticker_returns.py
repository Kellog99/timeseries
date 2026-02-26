from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Literal, Optional

import matplotlib.pyplot as plt
from fastapi import APIRouter, Depends
from fastapi.params import Query, Body
from tqdm import tqdm

from services.routers.load_data import get_data
from timeseries.metrics.returns import get_return
from .models.tickerdata import History, TickerReturn
from .utils import config_field

router = APIRouter(prefix="/advance_info")


@router.post("/returns", response_model=list[TickerReturn])
def get_tickers_return(
        tickers: list[str] = Body(...),
        path_data: Path | str = Depends(config_field("path_data")),
        save: bool = True
) -> list[TickerReturn]:
    """
    This function handles the retrival of multiple ticker at the same time.

    Args:
        tickers: list of the ticker to retrieve from the backend
        path_data: path to the stashed data
        save: flag that tells whether the data has to be stashed or not
    """
    out: list[TickerReturn] = []
    tickers_list = sorted(list(set(tickers)))
    if len(tickers_list)>0:
        pbar = tqdm(
            iterable=tickers_list,
            desc=f"Processing {tickers_list[0]}"
        )

        for ticker in pbar:
            pbar.set_description(f"Processing {ticker}")
            out.append(
                get_ticker_return(
                    ticker=ticker,
                    path_data=path_data,
                    save=save
                )
            )
    return out


@router.get("/return")
def get_ticker_return(
        ticker: str = Query(
            default=...,
            description="tickers to analyse."
        ),
        path_data: Path | str = Depends(config_field("path_data")),
        save: bool = Depends(config_field("save"))
) -> TickerReturn:
    """
    This function compute the ticker's percentage return for different type of time windows
    """
    today_str: str = datetime.today().strftime(format="%Y-%m-%d")
    today_date: date = datetime.strptime(today_str, "%Y-%m-%d").date()

    if today_date.weekday() > 4:
        # Now it is transformed in the closest saturday
        today_date -= timedelta(days=today_date.weekday() - 4)

    data: dict[str | date, History] = get_data(
        ticker=ticker,
        path_data=path_data,
        step_size=1,
        save=save
    ).history
    if not data:
        return TickerReturn(
            name=ticker,
            weeklyReturn=0,
            monthlyReturn=0,
            yearlyReturn=0,
            totalReturn=0
        )

    prices: dict[date, float] = {
        datetime.strptime(day, "%Y-%m-%d").date(): value.Close
        for day, value in data.items()
    }

    first_date = min(prices.keys())
    last_date = max(prices.keys())

    today_value = prices.get(today_date, prices[last_date])
    if today_value is None:
        today_value = prices.get(last_date)

    def get_daily_return(starting_date: date) -> float:

        if starting_date < first_date:
            starting_date = first_date
        starting_value = prices.get(starting_date, None)
        # This means that the date is either a weekend or an holiday
        # I search for the closest future date
        if starting_value is None:
            while starting_value is None:
                starting_date += timedelta(days=1)
                starting_value = prices.get(starting_date, None)

        return (today_value - starting_value) / starting_value * 100

    return TickerReturn(
        name=ticker,
        weeklyReturn=get_daily_return(starting_date=today_date - timedelta(weeks=1)),
        monthlyReturn=get_daily_return(starting_date=today_date - timedelta(weeks=4)),
        yearlyReturn=get_daily_return(starting_date=today_date - timedelta(days=365)),
        totalReturn=get_daily_return(starting_date=first_date)
    )


@router.get("/return_distribution")
def get_return_distribution(
        ticker: str = Query(
            default=...,
            descrption="tickers to analyse."
        ),
        path_data: Optional[Path | str] = Depends(config_field("path_data")),
        save: bool = Depends(config_field("save")),
        date_format: str = Depends(config_field("format")),
        scale: Literal["plain", "abs", "exp"] = Query(
            default=...,
            description=""
        ),
        bins: int = Depends(config_field("bins")),
        density: bool = True
) -> list[float]:
    """
    In this function it is created the histogram of the ticker's returns
    """
    if scale not in ["plain", "abs", "exp"]:
        raise ValueError("The chosen scale is not supported")
    ticker_history: dict[str | date, History] = get_data(
        ticker=ticker,
        path_data=path_data,
        save=save,
        step_size=1
    ).history
    history = {day: value.Close for day, value in ticker_history.items()}
    ticker_return: dict[date, float] = get_return(
        history=history,
        date_format=date_format,
        scale=scale
    )
    return list(ticker_return.values())
