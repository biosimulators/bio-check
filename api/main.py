import os
import logging
import uuid

import dotenv
from typing import *

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter
from pydantic import BeforeValidator
from starlette.middleware.cors import CORSMiddleware

# from bio_check import MONGO_URI
from data_model import DbClientResponse, UtcComparisonResult, UtcComparisonSubmission
from shared import save_uploaded_file, upload_blob
from shared import MongoDbConnector
from log_config import setup_logging

# --load env -- #

# dotenv.load_dotenv()


# -- constraints -- #

APP_TITLE = "bio-check"
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
MONGO_URI = os.getenv("MONGO_URI")
BUCKET_NAME = os.getenv("BUCKET_NAME")

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


# -- get mongo db -- #

db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id="service_requests")
app.mongo_client = db_connector.client

# It will be represented as a `str` on the model so that it can be serialized to JSON. Represents an ObjectId field in the database.
PyObjectId = Annotated[str, BeforeValidator(str)]


# -- initialization logic --

@app.on_event("startup")
def start_client() -> DbClientResponse:
    """TODO: generalize this to work with multiple client types. Currently using Mongo."""
    _time = db_connector.timestamp()
    try:
        app.mongo_client.admin.command('ping')
        msg = "Pinged your deployment. You successfully connected to MongoDB!"
    except Exception as e:
        msg = f"Failed to connect to MongoDB:\n{e}"
    return DbClientResponse(message=msg, db_type=DB_TYPE, timestamp=_time)


@app.on_event("shutdown")
def stop_mongo_client() -> DbClientResponse:
    _time = db_connector.timestamp()
    app.mongo_client.close()
    return DbClientResponse(message=f"{DB_TYPE} successfully closed!", db_type=DB_TYPE, timestamp=_time)


# -- endpoint logic -- #

@app.get("/")
def root():
    return {'bio-check-message': 'Hello from the Verification Service API!'}


# @app.post(
#     "/utc-comparison",  # "/biosimulators-utc-comparison",
#     response_model=UtcComparisonSubmission,
#     name="Uniform Time Course Comparison",
#     operation_id="utc-comparison",
#     summary="Compare UTC outputs from for a deterministic SBML model within a given archive.")
# async def utc_comparison(
#         uploaded_file: UploadFile = File(..., description="OMEX/COMBINE Archive File."),
#         simulators: List[str] = Query(default=["amici", "copasi", "tellurium"], description="List of simulators to compare"),
#         include_outputs: bool = Query(default=True, description="Whether to include the output data on which the comparison is based."),
#         comparison_id: Optional[str] = Query(default=None, description="Comparison ID to use."),
#         ground_truth_report: UploadFile = File(default=None, description="reports.h5 file defining the 'ground-truth' to be included in the comparison.")
#         ) -> UtcComparisonSubmission:
#     try:
#         job_id = str(uuid.uuid4())
#         _time = db_connector.timestamp()
#         save_dest = FILE_STORAGE_LOCATION  # tempfile.mktemp()  # TODO: replace with with S3 or google storage.
#
#         # TODO: remove this when using a shared filestore
#         if not os.path.exists(save_dest):
#             os.mkdir(save_dest)
#
#         # save uploaded omex file to shared storage
#         omex_fp = await save_uploaded_file(uploaded_file, save_dest)
#
#         # save uploaded reports file to shared storage if applicable
#         report_fp = await save_uploaded_file(ground_truth_report, save_dest) if ground_truth_report else None
#
#         pending_job_doc = await db_connector.insert_job_async(
#             collection_name="pending_jobs",
#             status="PENDING",
#             job_id=job_id,
#             omex_path=omex_fp,
#             simulators=simulators,
#             comparison_id=comparison_id or f"uniform-time-course-comparison-{job_id}",
#             timestamp=_time,
#             ground_truth_report_path=report_fp,
#             include_outputs=include_outputs)
#
#         # TODO: remove this when using shared file store.
#         # rmtree(save_dest)
#         return UtcComparisonSubmission(**pending_job_doc)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/utc-comparison",  # "/biosimulators-utc-comparison",
    response_model=UtcComparisonSubmission,
    name="Uniform Time Course Comparison",
    operation_id="utc-comparison",
    summary="Compare UTC outputs from for a deterministic SBML model within a given archive.")
async def utc_comparison(
        uploaded_file: UploadFile = File(..., description="OMEX/COMBINE Archive File."),
        simulators: List[str] = Query(default=["amici", "copasi", "tellurium"], description="List of simulators to compare"),
        include_outputs: bool = Query(default=True, description="Whether to include the output data on which the comparison is based."),
        comparison_id: Optional[str] = Query(default=None, description="Comparison ID to use."),
        ground_truth_report: UploadFile = File(default=None, description="reports.h5 file defining the 'ground-truth' to be included in the comparison.")
        ) -> UtcComparisonSubmission:
    try:
        # request specific params
        job_id = str(uuid.uuid4())
        _time = db_connector.timestamp()

        from tempfile import mkdtemp
        save_dest = mkdtemp()
        # bucket params
        upload_prefix = f"uploads/{job_id}/"
        bucket_prefix = f"gs://{BUCKET_NAME}/" + upload_prefix

        # Save uploaded omex file to Google Cloud Storage
        omex_fp = await save_uploaded_file(uploaded_file, save_dest)
        omex_blob_dest = upload_prefix + uploaded_file.filename
        upload_blob(BUCKET_NAME, omex_fp, omex_blob_dest)
        omex_path = bucket_prefix + uploaded_file.filename

        # Save uploaded reports file to Google Cloud Storage if applicable
        report_fp = None
        if ground_truth_report:
            report_fp = await save_uploaded_file(ground_truth_report, save_dest)
            report_blob_dest = upload_prefix + ground_truth_report.filename
            upload_blob(BUCKET_NAME, report_fp, report_blob_dest)
        report_path = bucket_prefix + ground_truth_report.filename if report_fp else None

        # run insert job
        pending_job_doc = await db_connector.insert_job_async(
            collection_name="pending_jobs",
            status="PENDING",
            job_id=job_id,
            omex_path=omex_path,
            simulators=simulators,
            comparison_id=comparison_id or f"uniform-time-course-comparison-{job_id}",
            timestamp=_time,
            ground_truth_report_path=report_path,
            include_outputs=include_outputs)

        # clean up local temp files
        os.remove(omex_fp)
        if report_fp:
            os.remove(report_fp)

        return UtcComparisonSubmission(**pending_job_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/fetch-results/{comparison_run_id}",
    response_model=UtcComparisonResult,
    operation_id='fetch-results',
    summary='Get the results of an existing uniform time course comparison.')
async def fetch_results(comparison_id: str):
    job = db_connector.fetch_job(comparison_id=comparison_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # assuming results are stored in the job document. TODO: what if this is not the case?
    job_id = job["job_id"]
    resp_content = {"job_id": job_id}
    key = "results" if job['status'] == 'COMPLETED' else "status"
    resp_content[key] = job[key]
    return UtcComparisonResult(content=resp_content)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


