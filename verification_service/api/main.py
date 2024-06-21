import os
import logging
import tempfile
import uuid
from shutil import rmtree

import dotenv
from typing import *
from datetime import datetime

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter, Body
from pydantic import BeforeValidator
from starlette.middleware.cors import CORSMiddleware
from pymongo.mongo_client import MongoClient

from verification_service.data_model import (
    UtcComparisonRequestParams,
    UtcComparison,
    Job,
    DbClientResponse,
    FetchResultsResponse,
    PendingJob)
from verification_service.api.handlers.io import save_uploaded_file
from verification_service.api.handlers.log_config import setup_logging
from verification_service.api.handlers.database import timestamp, insert_pending_job, fetch_comparison_job

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
DB_NAME = "service_requests"

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

# Represents an ObjectId field in the database.
# It will be represented as a `str` on the model so that it can be serialized to JSON.
PyObjectId = Annotated[str, BeforeValidator(str)]


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
    return {'verification-service-message': 'Hello from the Verification Service API!'}


@app.post(
    "/utc-comparison",  # "/biosimulators-utc-comparison",
    response_model=PendingJob,
    name="Biosimulator Uniform Time Course Comparison",
    operation_id="utc-comparison",
    summary="Compare UTC outputs from Biosimulators for a model from a given archive.")
async def utc_comparison(
        uploaded_file: UploadFile = File(..., description="OMEX/COMBINE Archive File."),
        simulators: List[str] = Query(default=["amici", "copasi", "tellurium"], description="List of simulators to compare"),
        include_output: bool = Query(default=True, description="Whether to include the output data on which the comparison is based."),
        comparison_id: Optional[str] = Query(default=None, description="Comparison ID to use."),
        ground_truth_report: UploadFile = File(default=None, description="reports.h5 file defining the so-called ground-truth to be included in the comparison.")
        ) -> PendingJob:
    try:
        job_id = str(uuid.uuid4())
        _time = timestamp()
        save_dest = "./tmp"  # tempfile.mktemp()  # TODO: replace with with S3 or google storage.

        # TODO: remove this when using a shared filestore
        if not os.path.exists(save_dest):
            os.mkdir(save_dest)

        # save uploaded omex file to shared storage
        omex_fp = await save_uploaded_file(uploaded_file, save_dest)

        # save uploaded reports file to shared storage if applicable
        report_fp = await save_uploaded_file(ground_truth_report, save_dest) if ground_truth_report else None

        job_doc = insert_pending_job(
            client=app.mongo_client,
            job_id=job_id,
            omex_path=omex_fp,
            simulators=simulators,
            comparison_id=comparison_id or f"uniform-time-course-comparison-{job_id}",
            timestamp=_time,
            reports_path=report_fp)

        # TODO: remove this when using shared file store.
        rmtree(save_dest)
        return PendingJob(**job_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/fetch-results/{comparison_run_id}",
    response_model=FetchResultsResponse,
    operation_id='fetch-results',
    summary='Get the results of an existing uniform time course comparison.')
async def fetch_results(comparison_run_id: str):
    job = fetch_comparison_job(client=app.mongo_client, job_id=comparison_run_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # assuming results are stored in the job document. TODO: what if this is not the case?
    resp_content = job['results'] if job['status'] == 'COMPLETED' else {"status": job['status']}
    return FetchResultsResponse(content=resp_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
