import json
import os

from fastapi import APIRouter
from fastapi.params import Query

from .models import Data, History

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
