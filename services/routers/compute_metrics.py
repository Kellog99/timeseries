import torch
from fastapi import APIRouter
from fastapi.params import Query

from metrics.ts_correlation import TimeSeriesCorrelation
from .load_data import get_data
from .models import History

router = APIRouter(prefix="/compute")


@router.get("/correlation")
def compute_correlation(
        tickers: list[str] = Query(
            default=...,
            description="List of tickers to compute the correlation with."
        )
) -> dict[str, list[list[float]]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "gpu")
    data: dict[str, dict[str, torch.Tensor]] = {}
    for ticker in tickers:
        hist: list[History] = get_data(
            ticker=ticker,
            step_size=0
        ).history
        data[ticker] = {
            daily.Date: torch.Tensor(daily.model_dump(exclude={"Date"}).values()) for daily in hist
        }
    cor_pearson = TimeSeriesCorrelation(reduction='mean', method='pearson')
    cor_spearman = TimeSeriesCorrelation(reduction='mean', method='spearman')
    out = {method: torch.eye(
        n=len(tickers),
        device=torch.device("cpu")
    ) for method in ["person", "spearman"]}

    for i in range(len(tickers)):
        data_i: dict = data[tickers[i]]
        dates_i: list[str] = data_i.keys()

        for j in range(i + 1, len(tickers)):
            data_j = data[tickers[j]]
            dates_j: list[str] = data_j.keys()
            intersection = list(set(dates_i) & set(dates_j))

            values_i = torch.concat([data_i[date] for date in intersection])
            values_j = torch.concat([data_j[date] for date in intersection])

            out["person"][i, j] = out["person"][i, j] = cor_pearson(values_i, values_j)
    return out
