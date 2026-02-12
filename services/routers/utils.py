import base64
from io import BytesIO
from pathlib import Path
from typing import TypeVar, Callable

import numpy as np
import torch
import torchmetrics
from fastapi import Request
from matplotlib import pyplot as plt
from torchmetrics import PearsonCorrCoef
from tqdm import tqdm

from .models.main_config import MainConfig

T = TypeVar("T")


############################### Depends decorator ###############################
def config_field(attr_name: str) -> Callable[..., T]:
    """
    A function for extracting an attribute from a config object in the app state.
    """

    def dependency(request: Request) -> T:
        config: MainConfig = request.app.state.config
        if not hasattr(config, attr_name):
            raise ValueError(f"Configuration attribute '{attr_name}' not found")
        return getattr(config, attr_name)

    dependency.__name__ = f"get_{attr_name}"
    return dependency


#################################################################################

def validate_path(path: str | Path, create: bool = True) -> Path | None:
    """
    validate the path that are passed through the whole application
    """
    if isinstance(path, (str, Path)):
        if isinstance(path, str):
            path: Path = Path(path).expanduser()
    else:
        raise ValueError(f"The path variable has an unsupported type, {type(path)}.")

    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def base64_plot_generator(matrix: np.ndarray):
    """
    This function create the plot associated to the correlation matrix and transform it into a base64 image.
    """
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


def compute_correlation_matrix(
        data: dict[str, dict],
        device: torch.device,
        metric_type: type[torchmetrics.Metric] = PearsonCorrCoef,
) -> np.ndarray:
    """
    This function has the role to compute the correlation matrix given
    1. the metric `metric_type`.
    2. the data
    Args:
        metric_type: the type of metric to compute
        data: this represents the dictionary of all the tickers that wants to be computed the correlation
        device: torch device
    Returns:
        np.ndarray of size len(tickers) x len(tickers)
    """

    matrix: np.ndarray = np.eye(len(data))
    tickers: list[str] = list(data.keys())
    metric = metric_type().to(device)

    for i, ticker_i in tqdm(enumerate(tickers), desc="Computing the correlation"):
        dates_i: set[str] = set(data[ticker_i].keys())
        for j, ticker_j in enumerate(tickers[i + 1:], start=i + 1):
            ############# Resetting the correlation functions #############
            metric.reset()
            ###############################################################
            dates_j: set[str] = set(data[ticker_j].keys())
            common_dates: list[str] = list(dates_i.intersection(dates_j))
            if len(common_dates) > 0:
                with torch.no_grad():
                    metric.update(
                        preds=torch.tensor([data[ticker_i][cd] for cd in common_dates]).to(device),
                        target=torch.tensor([data[ticker_j][cd] for cd in common_dates]).to(device)
                    )
                    corr: float = metric.compute()
                # save the results into the matrices
                matrix[i][j] = abs(corr)
                matrix[j][i] = abs(corr)

    return matrix
