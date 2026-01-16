import base64
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import Request
from matplotlib import pyplot as plt

from .models.main_config import MainConfig

############################### Depends decorator ###############################
attr_type = Literal[tuple(MainConfig.model_fields.keys())]


def config_field(attr_name: attr_type):
    """
    A function for extracting an attribute from a config object in the app state.
    """

    def dependency(request: Request):
        config: MainConfig = request.app.state.config
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
