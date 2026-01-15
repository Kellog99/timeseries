import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import api_router
from routers.models.main_config import MainConfig, parsed_argument

app = FastAPI(
    title='Financial analysis backend',
    description='This is the backend for the timeseries analysis.',
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Include routers
app.include_router(api_router)

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


@app.get("/")
def root():
    return {"message": "Welcome to the Finance Job API"}


if __name__ == "__main__":
    args = parsed_argument(MainConfig)
    config = MainConfig(**vars(args))
    app.state.config = config
    uvicorn.run(
        app,  # Use the app object directly instead of string
        host=config.host,
        port=config.port,
        workers=config.worker
    )
