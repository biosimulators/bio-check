import os
import logging
import uuid
import dotenv
from typing import *
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter
from starlette.middleware.cors import CORSMiddleware
from pymongo.mongo_client import MongoClient

from verification_service.data_model import (
    UtcComparisonRequest,
    UtcComparison,
    Job,
    DbClientResponse,
    FetchResultsResponse)
from verification_service.api.handlers.io import save_uploaded_file
from verification_service.api.handlers.log_config import setup_logging
from verification_service.api.handlers.database import timestamp


# --load env -- #

dotenv.load_dotenv()


# -- constraints -- #

APP_TITLE = "verification-service"
APP_VERSION = "0.0.1"

# TODO: update this
ORIGINS = [
    'http://127.0.0.1:8000',
    'http://127.0.0.1:4200',
    'http://127.0.0.1:4201',
    'http://127.0.0.1:4202',
    'http://localhost:4200',
    'http://localhost:4201',
    'http://localhost:4202',
    'https://biosimulators.org',
    'https://www.biosimulators.org',
    'https://biosimulators.dev',
    'https://www.biosimulators.dev',
    'https://run.biosimulations.dev',
    'https://run.biosimulations.org',
    'https://biosimulations.dev',
    'https://biosimulations.org',
    'https://bio.libretexts.org',
]

DB_TYPE = "mongo"  # ie: postgres, etc

MONGO_URI = os.getenv("MONGO_DB_URI")


# -- handle logging -- #

setup_logging()
logger = logging.getLogger(__name__)


# -- app components -- #

router = APIRouter()
app = FastAPI(title=APP_TITLE, version=APP_VERSION)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

app.mongo_client = MongoClient(MONGO_URI)


# -- initialization logic --

@app.on_event("startup")
def start_client() -> DbClientResponse:
    """TODO: generalize this to work with multiple client types. Currently using Mongo."""
    _time = timestamp()
    try:
        app.mongo_client.admin.command('ping')
        msg = "Pinged your deployment. You successfully connected to MongoDB!"
    except Exception as e:
        msg = f"Failed to connect to MongoDB:\n{e}"

    return DbClientResponse(message=msg, db_type=DB_TYPE, timestamp=_time)


@app.on_event("shutdown")
def stop_mongo_client() -> DbClientResponse:
    _time = timestamp()
    app.mongo_client.close()
    return DbClientResponse(message=f"{DB_TYPE} successfully closed!", db_type=DB_TYPE, timestamp=_time)


# -- endpoint logic -- #

@app.get("/")
def root():
    return {'verification-worker-message': 'Hello from the Verification Service API!'}


@app.post(
    "/utc-comparison",  # "/biosimulators-utc-comparison",
    response_model=Job,
    name="Biosimulator Uniform Time Course Comparison",
    operation_id="utc-comparison",
    summary="Compare UTC outputs from Biosimulators for a model from a given archive.")
async def utc_comparison(
        uploaded_file: UploadFile = File(..., description="OMEX/COMBINE Archive File."),
        # simulators: List[str] = Query(
        #     default=['amici', 'copasi', 'tellurium'],
        #     description="Simulators to include in the comparison."
        # ),
        # include_outputs: bool = Query(
        #     default=True,
        #     description="Whether to include the output data on which the comparison is based."
        # ),
        # comparison_id: str = Query(
        #     default=None,
        #     description="Descriptive identifier for this comparison."
        # ),
        comparison_params: UtcComparisonRequest = Query(..., description="Simulators to compare, whether to include output data, and descriptive id of comparison."),
        ground_truth_report: UploadFile = File(
            default=None,
            description="reports.h5 file defining the so-called ground-truth to be included in the comparison.")
        ) -> Job:
    job_id = str(uuid.uuid4())
    try:
        # save uploaded file to shared storage
        save_path = await save_uploaded_file(uploaded_file)

        pending_job_document = {
            "job_id": job_id,
            "status": "PENDING",
            "omex_path": save_path,
            "simulators": comparison_params.simulators,
            "comparison_id": comparison_params.comparison_id or f"uniform-time-course-comparison-{job_id}"
        }

        # create job record in MongoDB
        app.mongo_client.jobs.insert_one(pending_job_document)

        return Job(id=job_id, status="PENDING")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/fetch-results/{comparison_run_id}",
    response_model=FetchResultsResponse,
    operation_id='fetch-results',
    summary='Get the results of an existing uniform time course comparison.')
async def fetch_results(comparison_run_id: str):
    job = app.mongo_client.jobs.find_one({"job_id": comparison_run_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # assuming results are stored in the job document. TODO: what if this is not the case?
    resp_content = job['results'] if job['status'] == 'COMPLETED' else {"status": job['status']}

    return FetchResultsResponse(content=resp_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
