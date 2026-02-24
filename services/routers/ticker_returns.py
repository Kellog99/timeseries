from datetime import datetime, timedelta, date
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.params import Query, Body

from services.routers.load_data import get_data
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
    return [
        get_ticker_return(
            ticker=ticker,
            path_data=path_data,
            save=save
        ) for ticker in set(tickers)
    ]


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
    print(f" {ticker} ".center(50, "#"))
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
        print("starting date = ", starting_date)
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
