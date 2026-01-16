import argparse
from pathlib import Path
from typing import Optional

import torch
from pydantic import BaseModel, Field, model_validator


class MainConfig(BaseModel):
    ####################### CONFIG #######################
    port: int = Field(
        default=8000,
        description="Port number of the web server"
    )
    host: str = Field(
        default="localhost",
        description="Host of the web server"
    )
    worker: int = Field(
        default=1,
        description="Number of worker processes"
    )
    ####################### PATH #######################
    path_report: str = Field(
        default="~/Desktop/finance/report",
        description="Path to the folder where there are all the report about a specific portfolio benchmark"
    )
    path_data: str = Field(
        default="~/Desktop/finance/data/",
        description="Path to the data already downloaded."
    )
    ####################################################

    tickers: list[str] = Field(
        default=[
            "AOS", "ABT", "ABBV", "ACN", "ADBE", "AMD", "AES", "AFL", "A", "APD", "ABNB", "AKAM", "ALB", "ARE",
            "ALGN", "ALLE", "LNT", "ALL", "GOOGL", "GOOG", "MO", "AMZN", "AMCR", "AEE", "AEP", "AXP", "AIG", "AMT",
            "AWK", "AMP", "AME", "AMGN", "APH", "ADI", "AON", "APA", "APO", "AAPL", "AMAT", "APP", "APTV", "ACGL",
            "ADM", "ARES", "ANET", "AJG", "AIZ", "T", "ATO", "ADSK", "ADP", "AZO", "AVB", "AVY", "BKR",
            "BALL", "BAC", "BAX", "BDX", "BBY", "BIIB", "BLK", "BX", "XYZ", "BK", "BA", "BKNG",
            "BSX", "BMY", "AVGO", "BR", "BRO", "BLDR", "BG", "BXP", "CHRW", "CDNS", "CPT", "CPB", "COF"
        ],
        description="Tickers that have been selected."
    )
    save: bool = Field(
        default=True,
        description="It tells whether to save a ticker's data or not."
    )
    ####################### CHART #######################
    step_size: int = Field(
        default=200,
        description="Step size between points in the time series chart."
    )
    #####################################################
    device: str = Field(
        default="cuda" if torch.cuda.is_available() else "cpu",
        description="Device to run the operations on."
    )

    @model_validator(mode='after')
    def create_paths(self):
        """Create folders if they don't exist"""
        for field_name in ['path_report', 'path_data']:
            path_str = getattr(self, field_name)
            path = Path(path_str).expanduser()
            if not path.is_file() and not path.exists():
                path.mkdir(parents=True, exist_ok=True)
        return self


def parsed_argument(model_class) -> argparse.Namespace:
    """
    Add all fields from a Pydantic model to an argument parser
    """
    parser = argparse.ArgumentParser()

    for field_name, field_info in model_class.model_fields.items():
        default = field_info.default
        help_text = field_info.description or f"{field_name} parameter"
        field_type = field_info.annotation

        # Handle Optional types
        if hasattr(field_type, '__origin__') and field_type.__origin__ is type(Optional):
            field_type = field_type.__args__[0]
        parser.add_argument(
            f"--{field_name}",
            type=field_type,
            default=default,
            help=help_text
        )
    return parser.parse_args()
