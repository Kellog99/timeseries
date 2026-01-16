import heapq
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchmetrics
from fastapi import APIRouter, Depends
from fastapi.params import Query
from tqdm import tqdm

from .utils import validate_path, base64_plot_generator, config_field

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
print(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from timeseries.metrics.ts_correlation import PearsonCorrelation
from .load_data import get_data

router = APIRouter(prefix="/compute")


def get_correlation_matrix(
        metric_type: type[torchmetrics.Metric],
        path_to_correlation_matrix: Path | str,
        device: str | torch.device,
        tickers: list[str],
        path_data: Optional[Path],
        save: bool = True
) -> np.ndarray:
    # After this I am sure that device is torch.device
    if isinstance(device, (str, torch.device)):
        if isinstance(device, str):
            try:
                device = torch.device(device)
            except:
                device = torch.device("cpu")

    matrix: np.ndarray | None = None
    if path_to_correlation_matrix.exists():
        with open(path_to_correlation_matrix, "r") as f:
            correlation_matrix = json.load(f)

        # Use cache only if valid and tickers match
        if (correlation_matrix.get("matrix") and
                correlation_matrix.get("tickers") and
                correlation_matrix["tickers"] == tickers):
            matrix = np.array(correlation_matrix["matrix"])

    # Compute the actual correlation matrix if not cached or invalid
    if matrix is None:
        matrix: np.ndarray = np.eye(len(tickers))

        data: dict[str, dict] = {
            ticker: {
                dailyValue.Date: torch.Tensor(list(dailyValue.model_dump(exclude={"Date", "Volume"}).values()))
                for dailyValue in get_data(
                    path_data=path_data,
                    ticker=ticker,
                    step_size=1
                ).history
            }
            for ticker in tqdm(tickers, desc="loading the files")
        }

        for i, ticker_i in tqdm(enumerate(tickers), desc="Computing the correlation"):
            dates_i: set[str] = set(data[ticker_i].keys())
            for j, ticker_j in enumerate(tickers[i + 1:], start=i + 1):
                ################### Correlation functions ###################
                metric: torchmetrics.Metric = metric_type().to(device)
                #############################################################
                dates_j: set[str] = set(data[ticker_j].keys())
                common_dates: list[str] = list(dates_i.intersection(dates_j))

                if len(common_dates) > 0:
                    values_i = torch.stack(
                        [data[ticker_i].get(date) for date in common_dates]
                    )
                    values_j = torch.stack(
                        [data[ticker_j].get(date) for date in common_dates]
                    )
                    with torch.no_grad():
                        metric.update(values_i.to(device), values_j.to(device))
                        corr: float = metric.compute()

                    # save the results into the matrices
                    matrix[i][j] = corr
                    matrix[j][i] = corr

        if save:
            with open(path_to_correlation_matrix, "w") as f:
                json.dump({
                    "matrix": matrix.tolist(),
                    "tickers": tickers
                }, f)

    return matrix


@router.get("/correlation")
def get_correlation_matrix_plot(
        correlation: str = Query(
            default="pearson",
            description="Type of correlation to compute. Default: 'Pearson'",
        ),
        path_report: Path | str = Depends(config_field("path_report")),
        device: str = Depends(config_field("device")),
        tickers: list[str] = Depends(config_field("tickers")),
        path_data: Optional[Path] = Depends(config_field("path_data")),
        save: bool = Depends(config_field("save"))
) -> str:
    if correlation.lower() == "pearson":
        metric: type[torchmetrics.Metric] = PearsonCorrelation
    else:
        metric = None

    matrix: np.ndarray = get_correlation_matrix(
        metric_type=metric,
        path_to_correlation_matrix=validate_path(path_report) / f"{correlation}.json",
        device=device,
        tickers=tickers,
        path_data=validate_path(path_data),
        save=save
    )

    return base64_plot_generator(matrix)


@router.get("/corr_lead")
def get_leaderboard(
        k: int = Query(
            default=10,
            description="Number of element to show in the leaderboard."
        ),
        correlation: str = Query(
            default="pearson",
            description="Type of correlation from which the correlations are computed. Default: 'Pearson'",
        ),
        best: bool = Query(
            default=True,
            description="It tells whether the correlations are the best, True, or the worst, False."
        ),
        path_report: Path | str = Depends(config_field("path_report")),
        device: str = Depends(config_field("device")),
        tickers: list[str] = Depends(config_field("tickers")),
        path_data: Optional[Path] = Depends(config_field("path_data")),
        save: bool = Depends(config_field("save"))

) -> list[list[float | str]]:
    path_correlation_matrix = validate_path(path_report) / f"{correlation}.json"
    matrix = None
    if path_correlation_matrix.exists():
        with open(path_correlation_matrix, "r") as f:
            correlation_matrix = json.load(f)
        if (correlation_matrix.get("tickers") and
                correlation_matrix.get("tickers") == tickers):
            matrix = np.array(correlation_matrix.get("matrix"))

    if correlation.lower() == "pearson":
        metric: type[torchmetrics.Metric] = PearsonCorrelation
    else:
        metric = None

    if matrix is None:
        matrix = get_correlation_matrix(
            metric_type=metric,
            path_to_correlation_matrix=path_correlation_matrix,
            device=device,
            tickers=tickers,
            path_data=path_data,
            save=save
        )
    n = len(tickers)
    flat_with_idx: list[list[float | str]] = [
        [tickers[i], tickers[j], matrix[i, j]]
        for i in range(n)
        for j in range(n)
    ]
    k = min(k, len(flat_with_idx))
    if best:
        return heapq.nsmallest(k, flat_with_idx, key=lambda x: x[2])
    else:
        return heapq.nlargest(k, flat_with_idx, key=lambda x: x[2])
