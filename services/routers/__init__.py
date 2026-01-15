# API Router for all endpoints
from fastapi import APIRouter

# Import and include all sub-routers here
from .load_data import router as data_router
from .compute_metrics import router as metrics_router
api_router = APIRouter()

api_router.include_router(data_router)
api_router.include_router(metrics_router)
