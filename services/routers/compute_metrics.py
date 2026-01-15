import base64
import heapq
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from fastapi import APIRouter, Request
from fastapi.params import Query
from tqdm import tqdm

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
print(project_root)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from timeseries.metrics.ts_correlation import PearsonCorrelation
from .load_data import get_data
from .models.main_config import MainConfig

router = APIRouter(prefix="/compute")


def create_plot(matrix: np.ndarray):
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use imshow for matrix visualization with a colorbar
    im = ax.imshow(
        matrix,
        cmap='coolwarm',
        aspect='auto',
        vmin=0,
        vmax=1
    )
    # Add colorbar with white text
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelcolor='white', color='white')
    cbar.outline.set_edgecolor('white')

    # Remove axis if desired
    ax.set_axis_off()

    # Save plot to a BytesIO buffer
    buffer = BytesIO()
    plt.savefig(buffer,
                format='png',
                bbox_inches='tight',
                dpi=100,
                transparent=True
                )
    buffer.seek(0)
    plt.close(fig)  # Close the figure to free memory

    # Encode to base64
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"


@router.get("/correlation")
def get_correlation_matrix_service(
        request: Request
) -> str:
    config: MainConfig = request.app.state.config
    device = torch.device(config.device)

    path_report = Path(config.path_report).expanduser() / "correlation_matrix.json"
    matrix = None
    if path_report.exists():
        with open(path_report, "r") as f:
            correlation_matrix = json.load(f)

        # Use cache only if valid and tickers match
        if (correlation_matrix.get("matrix") and
                correlation_matrix.get("tickers") and
                correlation_matrix["tickers"] == config.tickers):
            matrix = np.array(correlation_matrix["matrix"])

    # Compute if not cached or invalid
    if matrix is None:
        matrix = get_correlation_matrix(
            tickers=config.tickers,
            path_data=Path(config.path_data).expanduser(),
            device=device
        )
        if config.save:
            with open(path_report, "w") as f:
                json.dump({
                    "matrix": matrix.tolist(),
                    "tickers": config.tickers
                }, f)
    return create_plot(matrix)


def get_correlation_matrix(
        tickers: list[str],
        path_data: Optional[Path] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
) -> np.ndarray:
    ##################### CORRELATION FUNCTIONS #####################
    out: np.ndarray = np.eye(len(tickers))
    #################################################################

    # Now I have to create
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
            corr_cross = PearsonCorrelation().to(device)
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
                    corr_cross.update(values_i.to(device), values_j.to(device))
                    corr: float = corr_cross.compute()

                # save the results into the matrices
                out[i][j] = corr
                out[j][i] = corr
    return np.array(out)


@router.get("/corr_lead")
def get_leaderboard(
        request: Request,
        k: int = Query(
            default=10,
            description="Number of element to show in the leaderboard."
        )
) -> list[list[float | str]]:
    config: MainConfig = request.app.state.config
    path_report = Path(config.path_report).expanduser() / "correlation_matrix.json"
    matrix = None
    if path_report.exists():
        with open(path_report, "r") as f:
            correlation_matrix = json.load(f)
        if correlation_matrix.get("tickers") and correlation_matrix.get("tickers") == config.tickers:
            matrix = np.array(correlation_matrix.get("matrix"))

    if matrix is None:
        matrix = get_correlation_matrix(
            tickers=config.tickers,
            path_data=Path(config.path_data).expanduser()
        )
    n = len(config.tickers)
    flat_with_idx: list[list[float | str]] = [
        [config.tickers[i], config.tickers[j], matrix[i, j]]
        for i in range(n)
        for j in range(n)
    ]
    k = min(k, len(flat_with_idx))
    return heapq.nsmallest(k, flat_with_idx, key=lambda x: x[2])
