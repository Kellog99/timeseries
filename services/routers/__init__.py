# API Router for all endpoints
from fastapi import APIRouter

# Import and include all sub-routers here
from .load_data import router as data_router

api_router = APIRouter()

api_router.include_router(data_router)
