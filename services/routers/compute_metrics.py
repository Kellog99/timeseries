import heapq
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchmetrics
from fastapi import APIRouter, Depends
from fastapi.params import Query
from torchmetrics import PearsonCorrCoef
from tqdm import tqdm

from .load_data import get_data
from .models.data import Data
from .utils import validate_path, base64_plot_generator, config_field, compute_correlation_matrix

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
print(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from timeseries.metrics.returns import get_daily_return

router = APIRouter(prefix="/compute")


def get_cached_correlation_matrix(
        path_report: Path,
        path_data: Optional[Path] = None,
        correlation: str = "pearson",
        device: str | torch.device = "cpu",
        tickers: Optional[list[str]] = None,
        save: bool = True,
) -> np.ndarray:
    """
    This function aims to handle all the loading of the information
    for executing the computation of the correlation matrix.
    """
    ############# Device #############
    if isinstance(device, (str, torch.device)):
        if isinstance(device, str):
            device = torch.device(device)
    else:
        raise ValueError(
            f"The type of the device, {type(device)}, is not supported."
        )
    ##################################

    ############# Correlation Metric #############
    metric: type[torchmetrics.Metric] | None = None
    if correlation.lower() == "pearson":
        metric: type[torchmetrics.Metric] = PearsonCorrCoef
    ##############################################

    matrix: np.ndarray | None = None
    #################### Checking if the matrix is cached ####################
    path_to_correlation_matrix: Path = validate_path(path_report) / f"{correlation}.json"
    if path_to_correlation_matrix.exists():
        with open(path_to_correlation_matrix, "r") as f:
            correlation_matrix = json.load(f)

        # Use cache only if valid and tickers match
        if (correlation_matrix.get("matrix") and
                correlation_matrix.get("tickers") and
                correlation_matrix["tickers"] == tickers):
            matrix = np.array(correlation_matrix["matrix"])
    ##########################################################################

    # Compute the actual correlation matrix if not cached or invalid
    if matrix is None:
        # If the time series are not given
        # They are upload from the device.
        if tickers is None:
            raise ValueError(
                "The matrix and the data are None. Hence the tickers are required for generating the matrix."
            )
        data: dict[str, dict[datetime, float]] = {}
        for ticker in tqdm(tickers, desc="loading the files"):
            # getting the history for a certain ticker
            ticker_data: Data = get_data(
                path_data=path_data,
                ticker=ticker,
                step_size=1
            )
            daily_return: list[tuple[datetime, float]] = get_daily_return(
                history=[(dv.Date, dv.Close) for dv in ticker_data.history]
            )
            data[ticker] = {date: day_return for [date, day_return] in daily_return}

        matrix: np.ndarray = compute_correlation_matrix(
            metric_type=metric,
            data=data,
            device=device,
        )

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
    matrix: np.ndarray = get_cached_correlation_matrix(
        correlation=correlation,
        path_report=path_report,
        tickers=tickers,
        device=device,
        path_data=path_data,
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
    """
    Given the correlation matrix, this function return the most or the less correlated indexes.
    """
    matrix: np.ndarray = get_cached_correlation_matrix(
        correlation=correlation,
        path_report=path_report,
        tickers=tickers,
        device=device,
        path_data=path_data,
        save=save
    )

    n = len(tickers)
    flat_with_idx: list[list[float | str]] = [
        [tickers[i], tickers[j], matrix[i, j]]
        for i in range(n)
        for j in range(n)
        if i != j
    ]
    k = min(k, len(flat_with_idx))
    if best:
        return heapq.nsmallest(k, flat_with_idx, key=lambda x: x[2])
    else:
        return heapq.nlargest(k, flat_with_idx, key=lambda x: x[2])


@router.get("/correlation_list")
def get_correlation(
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

) -> list[tuple[str, float]]:
    """
    Given the correlation matrix, this function return the most or the less correlated indexes.
    """
    matrix: np.ndarray = get_cached_correlation_matrix(
        correlation=correlation,
        path_report=path_report,
        tickers=tickers,
        device=device,
        path_data=path_data,
        save=save
    )

    # Removing the element in the diagonal
    matrix -= np.eye(len(matrix))

    n = len(tickers)
    flat_with_idx: list[tuple[str, float]] = [
        (tickers[i], matrix[i].max())
        for i in range(n)
    ]
    if best:
        return heapq.nsmallest(k, flat_with_idx, key=lambda x: x[1])
    else:
        return heapq.nlargest(k, flat_with_idx, key=lambda x: x[1])
