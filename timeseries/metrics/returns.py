import math
from datetime import datetime, date, timedelta


def get_return(
        history: dict[date | str, float],
        format: str = "%Y-%m-%d",
        window: date = timedelta(days=1),
) -> dict[date, float]:
    """
    This function compute the daily return of a certain list
    Args:
        history (dict[date | str, float]): this represents the univariate time series.
        format (str): the format of the history's data
        window
    Returns:
        list[tuple[datetime, float]]: this represents the daily return of a certain list
    """
    sorted_days: list[date] = sorted([
        day if isinstance(day, date) else datetime.strptime(day, format).date()
        for day in history.keys()
    ])

    out: dict[date, float] = {}
    if len(history) > 1:
        # Value of the daily value
        prev_value: float = history.get(sorted_days[0])

        for i in range(1, len(sorted_days)):
            # I compute the return in this two cases:
            # 1) the time difference is 1 day
            # 2) the new date is monday and the previous is friday, hence the time difference is not 1
            if (sorted_days[i] - sorted_days[i - 1] == timedelta(days=1) or
                    (sorted_days[i].weekday() == 0 and sorted_days[i - 1].weekday() == 4 and sorted_days[i] -
                     sorted_days[i - 1] == 3)):
                current_value: float = history.get(sorted_days[i])

                daily_return: float = (current_value - prev_value) / prev_value
                out[sorted_days[i]] = daily_return

            prev_value = history.get(sorted_days[i])
    return out


def daily_log_returns(
        history: list[tuple[datetime | str, float]],
        format: str = "%Y-%m-%d"
) -> list[tuple[datetime, float]]:
    """
    This function compute the daily log return of a certain list

    Args:
        history (list[tuple[datetime | str, float]]): this represents the univariate time series.
        format (str): the format of the history's data

    Returns:
        list[tuple[datetime, float]]: this represents the daily return of a certain list
    """
    daily_returns: list[tuple[datetime, float]] = get_return(history, format=format)
    return [(date, math.log(value)) for date, value in daily_returns]
