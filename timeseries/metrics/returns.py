import math
from datetime import datetime


def get_daily_return(
        history: list[tuple[datetime | str, float]],
        format: str = "%Y-%m-%d"
) -> list[tuple[datetime, float]]:
    """
    This function compute the daily return of a certain list
    Args:
        history (list[tuple[datetime | str, float]]): this represents the univariate time series.
        format (str): the format of the history's data
    Returns:
        list[tuple[datetime, float]]: this represents the daily return of a certain list
    """
    out = []
    if len(history) <= 1:
        return out
    else:
        # Value of the daily value
        prev_value: float = history[0][1]

        for i in range(1, len(history)):
            current_day = history[i][0]
            if isinstance(current_day, str):
                current_day: datetime = datetime.strptime(current_day, format)

            current_value: float = history[i][1]

            daily_return = (current_value - prev_value) / prev_value
            out.append((current_day, daily_return))
            prev_value = current_value
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
    daily_returns: list[tuple[datetime, float]] = get_daily_return(history, format=format)
    return [(date, math.log(value)) for date, value in daily_returns]
