import asyncio
import os
import logging
import uuid
import dotenv
from tempfile import mkdtemp
from typing import *

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter, Body, Request, Response
from pydantic import BeforeValidator
from starlette.middleware.cors import CORSMiddleware

from compose_api.compatible import COMPATIBLE_VERIFICATION_SIMS
# from bio_check import MONGO_URI
from data_model import DbClientResponse, UtcComparisonResult, PendingOmexJob, PendingSbmlJob, PendingSmoldynJob, CompatibleSimulators, Simulator, PendingUtcJob
from shared import upload_blob, MongoDbConnector, DB_NAME, DB_TYPE, BUCKET_NAME
from io_api import write_uploaded_file, save_uploaded_file, check_upload_file_extension
from log_config import setup_logging

# --load env -- #

dotenv.load_dotenv("../assets/.env_dev")


# -- constraints -- #

APP_TITLE = "bio-composer"
APP_VERSION = "1.0.0"

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


MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")


# -- handle logging -- #

# setup_logging()
# logger = logging.getLogger(__name__)


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

db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)
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
    return {'bio-composer-message': 'Hello from the BioComposer!'}


# run simulations

@app.post(
    "/run-smoldyn",  # "/biosimulators-utc-comparison",
    # response_model=PendingSmoldynJob,
    name="Run a smoldyn simulation",
    operation_id="run-smoldyn",
    tags=["Execute Simulations"],
    summary="Run a smoldyn simulation")
async def run_smoldyn(
        uploaded_file: UploadFile = File(..., description="Smoldyn Configuration File"),
        duration: int = Query(default=None, description="Simulation Duration"),
        dt: float = Query(default=None, description="Interval of step with which simulation runs"),
        # initial_molecule_state: List = Body(default=None, description="Mapping of species names to initial molecule conditions including counts and location.")
):
    try:
        job_id = "execute-simulations-" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        uploaded_file_location = await write_uploaded_file(job_id=job_id, uploaded_file=uploaded_file, bucket_name=BUCKET_NAME, extension='.txt')

        job_doc = {
            'job_id': job_id,
            'timestamp': _time,
            'status': 'PENDING',
            'path': uploaded_file_location,
            'duration': duration,
            'dt': dt,
            # 'initial_molecule_state': initial_molecule_state
        }
        pending_job = await db_connector.write(coll_name='pending_jobs', **job_doc)

        return job_doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/run-utc",  # "/biosimulators-utc-comparison",
    # response_model=PendingUtcJob,
    name="Run an ODE Uniform Time Course simulation",
    operation_id="run-utc",
    tags=["Execute Simulations"],
    summary="Run a Uniform Time Course simulation")
async def run_utc(
        uploaded_file: UploadFile = File(..., description="SBML File"),
        start: int = Query(..., description="Starting time for utc"),
        end: int = Query(..., description="Simulation Duration"),
        steps: int = Query(..., description="Number of points for utc"),
        simulator: str = Query(..., description="Simulator to use (one of: amici, copasi, tellurium, vcell)"),
):
    try:
        job_id = "execute-simulations-" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        uploaded_file_location = await write_uploaded_file(job_id=job_id, uploaded_file=uploaded_file, bucket_name=BUCKET_NAME, extension='.xml')

        job_doc = {
            'job_id': job_id,
            'timestamp': _time,
            'status': 'PENDING',
            'path': uploaded_file_location,
            'start': start,
            'end': end,
            'steps': steps,
            'simulator': simulator
        }
        pending_job = await db_connector.write(coll_name='pending_jobs', **job_doc)

        return job_doc
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# verification

@app.post(
    "/verify-omex",  # "/biosimulators-utc-comparison",
    response_model=PendingOmexJob,
    name="Uniform Time Course Comparison from OMEX/COMBINE archive",
    operation_id="verify-omex",
    tags=["Verification"],
    summary="Compare UTC outputs from a deterministic SBML model within an OMEX/COMBINE archive.")
async def verify_omex(
        uploaded_file: UploadFile = File(..., description="OMEX/COMBINE archive containing a deterministic SBML model"),
        simulators: List[str] = Query(default=["amici", "copasi", "tellurium"], description="List of simulators to compare"),
        include_outputs: bool = Query(default=True, description="Whether to include the output data on which the comparison is based."),
        selection_list: Optional[List[str]] = Query(default=None, description="List of observables to include in the return data."),
        comparison_id: Optional[str] = Query(default=None, description="Descriptive prefix to be added to this submission's job ID."),
        expected_results: UploadFile = File(default=None, description="reports.h5 file defining the expected results to be included in the comparison."),
        rTol: Optional[float] = Query(default=None, description="Relative tolerance to use for proximity comparison."),
        aTol: Optional[float] = Query(default=None, description="Absolute tolerance to use for proximity comparison.")
) -> PendingOmexJob:
    try:
        # request specific params
        if comparison_id is None:
            compare_id = "utc_comparison_omex"
        else:
            compare_id = comparison_id

        job_id = "verification-" + compare_id + "-" + str(uuid.uuid4())
        _time = db_connector.timestamp()

        # bucket params
        upload_prefix = f"uploads/{job_id}/"
        bucket_prefix = f"gs://{BUCKET_NAME}/" + upload_prefix

        save_dest = mkdtemp()

        fp = await save_uploaded_file(uploaded_file, save_dest)  # save uploaded file to ephemeral store

        # Save uploaded omex file to Google Cloud Storage
        uploaded_file_location = None
        properly_formatted_omex = check_upload_file_extension(uploaded_file, 'uploaded_file', '.omex')
        if properly_formatted_omex:
            blob_dest = upload_prefix + fp.split("/")[-1]
            upload_blob(bucket_name=BUCKET_NAME, source_file_name=fp, destination_blob_name=blob_dest)
            uploaded_file_location = blob_dest

        # Save uploaded reports file to Google Cloud Storage if applicable
        report_fp = None
        report_blob_dest = None
        if expected_results:
            # handle incorrect files upload
            properly_formatted_report = check_upload_file_extension(expected_results, 'expected_results', '.h5')
            if properly_formatted_report:
                report_fp = await save_uploaded_file(expected_results, save_dest)
                report_blob_dest = upload_prefix + report_fp.split("/")[-1]
            upload_blob(bucket_name=BUCKET_NAME, source_file_name=report_fp, destination_blob_name=report_blob_dest)
        report_location = report_blob_dest

        pending_job_doc = await db_connector.insert_job_async(
            collection_name="pending_jobs",
            status="PENDING",
            job_id=job_id,
            path=uploaded_file_location,
            simulators=simulators,
            timestamp=_time,
            expected_results=report_location,
            include_outputs=include_outputs,
            rTol=rTol,
            aTol=aTol,
            selection_list=selection_list
        )

        # clean up local temp files
        os.remove(fp)
        if report_fp:
            os.remove(report_fp)

        return PendingOmexJob(**pending_job_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/verify-sbml",
    response_model=PendingSbmlJob,
    name="Uniform Time Course Comparison from SBML file",
    operation_id="verify-sbml",
    tags=["Verification"],
    summary="Compare UTC outputs from a deterministic SBML model.")
async def verify_sbml(
        uploaded_file: UploadFile = File(..., description="A deterministic SBML model."),
        start: int = Query(..., description="Start time of the simulation (output start time)"),
        end: int = Query(..., description="End time of simulation (end)"),
        steps: int = Query(..., description="Number of simulation steps to run"),
        simulators: List[str] = Query(default=["copasi", "tellurium"], description="List of simulators to compare"),
        include_outputs: bool = Query(default=True, description="Whether to include the output data on which the comparison is based."),
        comparison_id: Optional[str] = Query(default=None, description="Descriptive prefix to be added to this submission's job ID."),
        expected_results: UploadFile = File(default=None, description="reports.h5 file defining the expected results to be included in the comparison."),
        rTol: Optional[float] = Query(default=None, description="Relative tolerance to use for proximity comparison."),
        aTol: Optional[float] = Query(default=None, description="Absolute tolerance to use for proximity comparison."),
        selection_list: Optional[List[str]] = Query(default=None, description="List of observables to include in the return data."),
) -> PendingSbmlJob:
    try:
        # request specific params
        if comparison_id is None:
            compare_id = "utc_comparison_omex"
        else:
            compare_id = comparison_id

        job_id = "verification-" + compare_id + "-" + str(uuid.uuid4())
        _time = db_connector.timestamp()

        # bucket params
        upload_prefix = f"uploads/{job_id}/"
        bucket_prefix = f"gs://{BUCKET_NAME}/" + upload_prefix

        save_dest = mkdtemp()

        fp = await save_uploaded_file(uploaded_file, save_dest)  # save uploaded file to ephemeral store

        # Save uploaded omex file to Google Cloud Storage
        uploaded_file_location = None
        properly_formatted_sbml = check_upload_file_extension(uploaded_file, 'uploaded_file', '.xml')
        if properly_formatted_sbml:
            blob_dest = upload_prefix + fp.split("/")[-1]
            upload_blob(bucket_name=BUCKET_NAME, source_file_name=fp, destination_blob_name=blob_dest)
            uploaded_file_location = blob_dest

        # Save uploaded reports file to Google Cloud Storage if applicable
        report_fp = None
        report_blob_dest = None
        if expected_results:
            # handle incorrect files upload
            properly_formatted_report = check_upload_file_extension(expected_results, 'expected_results', '.h5')
            if properly_formatted_report:
                report_fp = await save_uploaded_file(expected_results, save_dest)
                report_blob_dest = upload_prefix + report_fp.split("/")[-1]
            upload_blob(bucket_name=BUCKET_NAME, source_file_name=report_fp, destination_blob_name=report_blob_dest)
        report_location = report_blob_dest

        pending_job_doc = await db_connector.insert_job_async(
            collection_name="pending_jobs",
            status="PENDING",
            job_id=job_id,
            comparison_id=compare_id,
            path=uploaded_file_location,
            simulators=simulators,
            timestamp=_time,
            start=start,
            end=end,
            steps=steps,
            include_outputs=include_outputs,
            expected_results=report_location,
            rTol=rTol,
            aTol=aTol,
            selection_list=selection_list
        )

        # clean up local temp files
        os.remove(fp)

        return PendingSbmlJob(**pending_job_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/get-output/{job_id}",
    response_model=UtcComparisonResult,
    operation_id='get-verify-output',
    tags=["Data"],
    summary='Get the results of an existing simulation run.')
async def fetch_results(job_id: str) -> UtcComparisonResult:
    colls = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
    for collection in colls:
        job = db_connector.db[collection].find_one({'job_id': job_id})
        if not isinstance(job, type(None)):
            job.pop('_id')
            return UtcComparisonResult(content=job)

    raise HTTPException(status_code=404, detail="Comparison not found")


@app.post(
    "/get-compatible-for-verification",
    response_model=CompatibleSimulators,
    operation_id='get-compatible-for-verification',
    tags=["Data"],
    summary='Get the simulators that are compatible with either a given OMEX/COMBINE archive or SBML model simulation.')
async def get_compatible_for_verification(
        uploaded_file: UploadFile = File(..., description="Either a COMBINE/OMEX archive or SBML file to be simulated."),
        versions: bool = Query(default=False, description="Whether to include the simulator versions for each compatible simulator."),
) -> CompatibleSimulators:
    try:
        filename = uploaded_file.filename
        simulators = COMPATIBLE_VERIFICATION_SIMS.copy()  # TODO: dynamically extract this!

        # handle filetype: amici is not compatible with sbml verification at the moment
        if not filename.endswith(".omex"):
            for sim in simulators:
                if sim[0] == 'amici':
                    simulators.remove(sim)

        compatible_sims = []
        for data in simulators:
            name = data[0]
            version = data[1]
            sim = Simulator(name=name, version=version if versions is not False else None)
            compatible_sims.append(sim)

        return CompatibleSimulators(file=uploaded_file.filename, simulators=compatible_sims)
    except Exception as e:
        raise HTTPException(status_code=404, detail="Comparison not found")


# TODO: Implement smoldyn simularium execution!


# TODO: eventually implement this
# @app.get("/events/{job_id}", status_code=200)
# async def get_job_events(request: Request, job_id: str):
#     async def event_generator():
#         while True:
#             job = db_connector.fetch_job(job_id=job_id)
#             if job['status'] in ['completed', 'cancelled']:
#                 yield f"data: {job['status']}\n\n"
#                 break
#             yield f"data: {job['status']}\n\n"
#             await asyncio.sleep(5)
#     return Response(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


