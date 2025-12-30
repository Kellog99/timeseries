import json
import os

from fastapi import APIRouter
from fastapi.params import Query

from .models import Data, History, Portfolio

router = APIRouter(prefix="/load")


@router.get("/data")
def get_data(
        ts_repo: str = Query(
            default=...,
            description="It represent the path to validate."
        )
) -> list[Data]:
    out: list[Data] = []
    for file in os.listdir(ts_repo):
        if "json" in file:
            with open(os.path.join(ts_repo, file), "r") as f:
                data_json = json.load(f)
            out.append(
                Data(
                    name=data_json["name"],
                    description=data_json["description"],
                    history={
                        date: History(**history)
                        for date, history in data_json["history"].items()
                    }
                )
            )
    return out


@router.get("/portfolio")
def get_portfolio(
        portfolio_path: str = Query(
            default=...,
            description="It represent the path to the portfolio time series."
        )
) -> Portfolio:
    if os.path.exists(portfolio_path):
        with open(portfolio_path, "r") as f:
            data_json = json.load(f)
        return Portfolio.model_validate_json(data_json)
    else:
        raise ValueError("The path to the portfolio analysis does not exists")
