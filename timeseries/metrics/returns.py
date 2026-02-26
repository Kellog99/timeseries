import math
from datetime import datetime, date, timedelta
from typing import Literal


def get_return(
        history: dict[str | date, float],
        date_format: str = "%Y-%m-%d",
        scale: Literal["plain", "abs", "exp"] = "plain",
) -> dict[date, float]:
    """
    This function compute the daily return of a certain list
    Args:
        history (dict[date | str, float]): this represents the univariate time series.
        date_format (str): the format of the history's data
        scale (Literal["plain","log", "exponential"]): it represents the scale where the return lies
    Returns:
        list[tuple[datetime, float]]: this represents the daily return of a certain list
    """
    sorted_days: list[date] = sorted([
        day if isinstance(day, date) else datetime.strptime(day, date_format).date()
        for day in history.keys()
    ])
    # I have to make sure to have the keys in a certain format
    history: dict[date, float] = {
        key if isinstance(key, date) else datetime.strptime(key, date_format).date(): value
        for key, value in history.items()
    }

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

                ############################ Scale of the return ############################
                daily_return: float = (current_value - prev_value) / prev_value

                match scale:
                    case "plain":
                        daily_return = daily_return
                    case "abs":
                        daily_return: float = abs(daily_return)
                    case "exp":
                        daily_return: float = math.exp(daily_return)
                #############################################################################

                out[sorted_days[i]] = daily_return

            prev_value = history.get(sorted_days[i])
    return out
