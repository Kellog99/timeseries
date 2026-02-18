from datetime import datetime, timedelta, date
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.params import Query, Body

from services.routers.load_data import get_data
from .models.data import History, TickerReturn
from .utils import config_field

router = APIRouter(prefix="/advance_info")


@router.post("/returns", response_model=list[TickerReturn])
def get_returns(
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
    return [
        get_return(ticker, path_data, save)
        for ticker in tickers
    ]


@router.get("/return")
def get_return(
        ticker: str = Query(
            default=...,
            description="tickers to analyse."
        ),
        path_data: Path | str = Depends(config_field("path_data")),
        save: bool = True
) -> TickerReturn:
    """
    This function compute the ticker's percentage return for different type of time windows
    """
    today_str: str = datetime.today().strftime(format="%Y-%m-%d")
    today_date: date = datetime.strptime(today_str, "%Y-%m-%d").date()

    if today_date.weekday() > 4:
        # Now it is transformed in the closest saturday
        today_date -= timedelta(days=today_date.weekday() - 4)

    data: list[History] = get_data(
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
        datetime.strptime(h.Date, "%Y-%m-%d").date(): h.Close
        for h in data
    }

    first_date = min(prices)
    last_date = max(prices)

    today_value = prices.get(today_date, prices[last_date])
    if today_value is None:
        today_value = prices.get(last_date)

    def get_daily_return(starting_date: date) -> float:

        if starting_date < first_date:
            starting_date = first_date
        print("starting date = ", starting_date)
        starting_value = prices.get(starting_date, None)
        if starting_value is None:
            return 0

        return (today_value - starting_value) / starting_value * 100

    return TickerReturn(
        name=ticker,
        weeklyReturn=get_daily_return(starting_date=today_date - timedelta(weeks=1)),
        monthlyReturn=get_daily_return(starting_date=today_date - timedelta(weeks=4)),
        yearlyReturn=get_daily_return(starting_date=today_date - timedelta(days=52)),
        totalReturn=get_daily_return(starting_date=first_date)
    )
