import argparse
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import api_router

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
    return {"message": "Welcome to the TITANN Job API"}


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ts_storage", "-ds",
        type=str,
        default="./time_series_repo",
        help="Path to internal storage of timeseries."
    )

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0", help="Host to bind the server to (default: localhost)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Uvicorn worker processes"
    )
    parser.add_argument(
        '--benchmark_output_dir',
        type=str,
        default='benchmark_results',
        help='Path to benchmark output directory'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    uvicorn.run(
        app,  # Use the app object directly instead of string
        host=args.host,
        port=args.port,
        workers=args.workers
    )
