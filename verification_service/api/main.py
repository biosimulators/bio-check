import logging
import uuid
from typing import *

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter
from starlette.middleware.cors import CORSMiddleware
import pymongo as db

from verification_service.data_model import UtcComparisonRequest, UtcComparison, PendingJob, Job
from verification_service.api.handlers.io import save_uploaded_file
from verification_service.api.handlers.log_config import setup_logging


setup_logging()

logger = logging.getLogger(__name__)


app = FastAPI(title='verification-service', version='1.0.0')
router = APIRouter()

origins = [
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        db.jobs.insert_one(pending_job_document)

        return Job(id=job_id, status="PENDING")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/fetch-results/{job_id}", response_model=UtcComparison)
async def fetch_results(job_id: str):
    job = db.jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job['status'] == 'COMPLETED':
        # Assuming results are stored in the job document
        return job['results']
    else:
        return {"status": job['status']}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
