"""Returns between consecutive observed sessions, including exchange holidays."""
import math
from datetime import date, datetime
from typing import Literal


def get_return(
    history: dict[str | date, float],
    date_format: str = "%Y-%m-%d",
    scale: Literal["plain", "abs", "exp"] = "plain"
) -> dict[date, float]:
    """Decimal simple returns; abs and exp are display transforms, not log returns."""
    if scale not in {"plain", "abs", "exp"}:
        raise ValueError("Unknown return scale.")
    prices = {}
    for day, value in history.items():
        key = day if isinstance(day, date) else datetime.strptime(day, date_format).date()
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Prices must be positive and finite.")
        if key in prices:
            raise ValueError("Duplicate observation date.")
        prices[key] = value
    days = sorted(prices)
    result = {}
    for previous, current in zip(days, days[1:]):
        value = prices[current] / prices[previous] - 1
        result[current] = abs(value) if scale == "abs" else math.exp(value) if scale == "exp" else value
    return result
