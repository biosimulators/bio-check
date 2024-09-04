import asyncio
import os
import logging
import uuid
from enum import Enum

import dotenv
from tempfile import mkdtemp
from typing import *

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter, Body, Request, Response
from fastapi.responses import FileResponse
from pydantic import BeforeValidator, Field
from starlette.middleware.cors import CORSMiddleware

from compatible import COMPATIBLE_VERIFICATION_SIMULATORS
# from bio_check import MONGO_URI
from data_model import DbClientResponse, UtcComparisonResult, PendingSmoldynJob, CompatibleSimulators, Simulator, PendingUtcJob, OutputData, PendingSimulariumJob, CompositionSpecification, PendingSbmlVerificationJob, PendingOmexVerificationJob, PendingCompositionJob, AgentParameters
from shared import upload_blob, MongoDbConnector, DB_NAME, DB_TYPE, BUCKET_NAME, JobStatus, DatabaseCollections, file_upload_prefix, BaseModel
from io_api import write_uploaded_file, save_uploaded_file, check_upload_file_extension, download_file_from_bucket
from log_config import setup_logging

# --load env -- #

dotenv.load_dotenv("../assets/.env_dev")


# -- constraints -- #

APP_TITLE = "bio-compose"
APP_VERSION = "0.1.0"

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
    return {'bio-compose-api-endpoint-root': 'https://biochecknet.biosimulations.org'}


# run simulations

@app.post(
    "/run-smoldyn",  # "/biosimulators-utc-comparison",
    # response_model=PendingSmoldynJob,
    name="Run a smoldyn simulation",
    operation_id="run-smoldyn",
    tags=["Simulation Execution"],
    summary="Run a smoldyn simulation.")
async def run_smoldyn(
        uploaded_file: UploadFile = File(..., description="Smoldyn Configuration File"),
        duration: int = Query(default=None, description="Simulation Duration"),
        dt: float = Query(default=None, description="Interval of step with which simulation runs"),
        # initial_molecule_state: List = Body(default=None, description="Mapping of species names to initial molecule conditions including counts and location.")
):
    try:
        job_id = "simulation-execution-smoldyn" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        uploaded_file_location = await write_uploaded_file(job_id=job_id, uploaded_file=uploaded_file, bucket_name=BUCKET_NAME, extension='.txt')

        pending_job = await db_connector.insert_job_async(
            collection_name=DatabaseCollections.PENDING_JOBS.value,
            job_id=job_id,
            timestamp=_time,
            status=JobStatus.PENDING.value,
            path=uploaded_file_location,
            duration=duration,
            dt=dt
        )

        return pending_job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/run-utc",  # "/biosimulators-utc-comparison",
    # response_model=PendingUtcJob,
    name="Run an ODE Uniform Time Course simulation",
    operation_id="run-utc",
    tags=["Simulation Execution"],
    summary="Run a Uniform Time Course simulation.")
async def run_utc(
        uploaded_file: UploadFile = File(..., description="SBML File"),
        start: int = Query(..., description="Starting time for utc"),
        end: int = Query(..., description="Simulation Duration"),
        steps: int = Query(..., description="Number of points for utc"),
        simulator: str = Query(..., description="Simulator to use (one of: amici, copasi, tellurium, vcell)"),
):
    try:
        job_id = "simulation-execution-utc" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        uploaded_file_location = await write_uploaded_file(job_id=job_id, uploaded_file=uploaded_file, bucket_name=BUCKET_NAME, extension='.xml')

        pending_job = await db_connector.insert_job_async(
            collection_name=DatabaseCollections.PENDING_JOBS.value,
            job_id=job_id,
            timestamp=_time,
            status=JobStatus.PENDING.value,
            path=uploaded_file_location,
            start=start,
            end=end,
            steps=steps,
            simulator=simulator
        )

        return pending_job
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# verification

@app.post(
    "/verify-omex",  # "/biosimulators-utc-comparison",
    response_model=PendingOmexVerificationJob,
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
) -> PendingOmexVerificationJob:
    try:
        # request specific params
        if comparison_id is None:
            compare_id = "utc_comparison_omex"
        else:
            compare_id = comparison_id

        job_id = "verification-" + compare_id + "-" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        upload_prefix, bucket_prefix = file_upload_prefix(job_id)
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
            collection_name=DatabaseCollections.PENDING_JOBS.value,
            status=JobStatus.PENDING.value,
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

        return PendingOmexVerificationJob(**pending_job_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/verify-sbml",
    response_model=PendingSbmlVerificationJob,
    name="Uniform Time Course Comparison from SBML file",
    operation_id="verify-sbml",
    tags=["Verification"],
    summary="Compare UTC outputs from a deterministic SBML model.")
async def verify_sbml(
        uploaded_file: UploadFile = File(..., description="A deterministic SBML model."),
        start: int = Query(..., description="Start time of the simulation (output start time)"),
        end: int = Query(..., description="End time of simulation (end)"),
        steps: int = Query(..., description="Number of simulation steps to run"),
        simulators: List[str] = Query(default=["amici", "copasi", "tellurium"], description="List of simulators to compare"),
        include_outputs: bool = Query(default=True, description="Whether to include the output data on which the comparison is based."),
        comparison_id: Optional[str] = Query(default=None, description="Descriptive prefix to be added to this submission's job ID."),
        expected_results: UploadFile = File(default=None, description="reports.h5 file defining the expected results to be included in the comparison."),
        rTol: Optional[float] = Query(default=None, description="Relative tolerance to use for proximity comparison."),
        aTol: Optional[float] = Query(default=None, description="Absolute tolerance to use for proximity comparison."),
        selection_list: Optional[List[str]] = Query(default=None, description="List of observables to include in the return data."),
) -> PendingSbmlVerificationJob:
    try:
        # request specific params
        if comparison_id is None:
            compare_id = "utc_comparison_sbml"
        else:
            compare_id = comparison_id

        job_id = "verification-" + compare_id + "-" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        upload_prefix, bucket_prefix = file_upload_prefix(job_id)
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
            collection_name=DatabaseCollections.PENDING_JOBS.value,
            status=JobStatus.PENDING.value,
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

        return PendingSbmlVerificationJob(**pending_job_doc)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/get-process-bigraph-addresses",
    operation_id="get-process-bigraph-addresses",
    tags=["Composition"],
    summary="Get process bigraph implementation addresses for composition specifications.")
def get_process_bigraph_addresses() -> List[str]:
    try:
        from biosimulators_processes import CORE
        return list(CORE.process_registry.registry.keys())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/run-composition",
    # response_model=PendingCompositionJob,
    operation_id='run-composition',
    tags=["Composition"],
    summary='Run a composite simulation.')
async def run_composition(
        source: UploadFile = File(..., description="Upload source file"),
        composition_spec: CompositionSpecification = Body(..., description="ProcessBigraph-compliant specification of composition."),
        duration: int = Query(..., description="Duration of the simulation in seconds."),
):
    try:
        # job params
        job_id = "composition-run" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        if composition_spec.composition_id is None:
            composition_spec.composition_id = job_id

        # insert a config with source (currently only supporting UTC MODEL CONFIG) TODO: expand this
        upload_prefix, bucket_prefix = file_upload_prefix(job_id)
        save_dest = mkdtemp()
        fp = await save_uploaded_file(source, save_dest)  # save uploaded file to ephemeral store

        # Save uploaded omex file to Google Cloud Storage
        uploaded_file_location = None
        properly_formatted_sbml = check_upload_file_extension(source, 'uploaded_file', '.xml')
        if properly_formatted_sbml:
            blob_dest = upload_prefix + fp.split("/")[-1]
            upload_blob(bucket_name=BUCKET_NAME, source_file_name=fp, destination_blob_name=blob_dest)
            uploaded_file_location = blob_dest

        # format process bigraph spec needed for Composite()
        spec = {}
        for node in composition_spec.nodes:
            name = node.name
            node_spec = node.model_dump()
            node_spec.pop("name")
            node_spec.pop("node_type")
            node_spec["_type"] = node.node_type
            spec[name] = node_spec

            if 'emitter' not in node.address:
                spec[name]['config'] = {
                    'model': {
                        'model_source': uploaded_file_location
                    }
                }

        # write job as dict to db
        # job = PendingCompositionJob(composition=spec, duration=duration, timestamp=_time, job_id=job_id)
        job = {'composition': spec, 'duration': duration, 'timestamp': _time, 'job_id': job_id}
        await db_connector.insert_job_async(collection_name=DatabaseCollections.PENDING_JOBS.value, **job)

        return job
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# TODO: allow this for admins
# @app.get("/get-jobs")
# async def get_jobs(collection: str = Query(...)):
#     return [job['job_id'] for job in db_connector.db[collection].find()]


@app.get(
    "/get-output/{job_id}",
    # response_model=OutputData,
    operation_id='get-verify-output',
    tags=["Data"],
    summary='Get the results of an existing simulation run.')
async def fetch_results(job_id: str):
    # TODO: refactor this!

    # state-case: job is completed
    job = await db_connector.read(collection_name="completed_jobs", job_id=job_id)

    # state-case: job has failed
    if job is None:
        job = await db_connector.read(collection_name="failed_jobs", job_id=job_id)

    # state-case: job is not in completed:
    if job is None:
        job = await db_connector.read(collection_name="in_progress_jobs", job_id=job_id)

    # state-case: job is not in progress:
    if job is None:
        job = await db_connector.read(collection_name="pending_jobs", job_id=job_id)

    # return-case: job exists as either completed, failed, in_progress, or pending
    if not isinstance(job, type(None)):
        # remove autogen obj
        job.pop('_id')

        # status/content-case: case: job is completed
        if job['status'] == "COMPLETED":
            # check output for type (either raw data or file download)
            job_data = job['results'].get('results') or job['results']

            # job has results
            if job_data is not None:
                remote_fp = None

                # output-type-case: output is saved as a dict
                if isinstance(job_data, dict):
                    # output-case: output content in dict is a downloadable file
                    if "results_file" in job_data.keys():
                        remote_fp = job_data['results_file']
                    # status/output-case: job is complete and output content is raw data and so return the full data TODO: do something better here
                    else:
                        return OutputData(content=job)

                # output-type-case: output is saved as flattened (str) and thus also a file download
                elif isinstance(job_data, str):
                    remote_fp = job_data

                # content-case: output content relates to file download
                if remote_fp is not None:
                    temp_dest = mkdtemp()
                    local_fp = download_file_from_bucket(source_blob_path=remote_fp, out_dir=temp_dest, bucket_name=BUCKET_NAME)

                    return FileResponse(path=local_fp, media_type="application/octet-stream", filename=local_fp.split("/")[-1])

        # status/content-case: job is either pending or in progress and does not contain files to download
        else:
            # acknowledge the user submission to differentiate between original submission
            status = job['status']
            job['status'] = 'SUBMITTED:' + status

            return OutputData(content=job)

    # return-case: no job exists in any collection by that id
    else:
        raise HTTPException(status_code=404, detail="Comparison not found")


@app.post(
    "/generate-simularium-file",
    # response_model=PendingSimulariumJob,
    operation_id='generate-simularium-file',
    tags=["Files"],
    summary='Generate a simularium file with a compatible simulation results file from either Smoldyn, SpringSaLaD, or ReaDDy.')
async def generate_simularium_file(
        uploaded_file: UploadFile = File(..., description="A file containing results that can be parse by Simularium (spatial)."),
        box_size: float = Query(..., description="Size of the simulation box as a floating point number."),
        filename: str = Query(default=None, description="Name desired for the simularium file. NOTE: pass only the file name without an extension."),
        translate_output: bool = Query(default=True, description="Whether to translate the output trajectory prior to converting to simularium. See simulariumio documentation for more details."),
        validate_output: bool = Query(default=True, description="Whether to validate the outputs for the simularium file. See simulariumio documentation for more details."),
        agent_parameters: AgentParameters = Body(default=None, description="Parameters for the simularium agents defining either radius or mass and density.")
):
    job_id = "files-generate-simularium-file" + str(uuid.uuid4())
    _time = db_connector.timestamp()
    upload_prefix, bucket_prefix = file_upload_prefix(job_id)
    uploaded_file_location = await write_uploaded_file(job_id=job_id, uploaded_file=uploaded_file, bucket_name=BUCKET_NAME, extension='.txt')

    # new simularium job in db
    if filename is None:
        filename = 'simulation'

    agent_params = {}
    if agent_parameters is not None:
        for agent_param in agent_parameters.agents:
            agent_params[agent_param.name] = agent_param.model_dump()

    new_job_submission = await db_connector.insert_job_async(
        collection_name=DatabaseCollections.PENDING_JOBS.value,
        status=JobStatus.PENDING.value,
        job_id=job_id,
        timestamp=_time,
        path=uploaded_file_location,
        filename=filename,
        box_size=box_size,
        translate_output=translate_output,
        validate_output=validate_output,
        agent_parameters=agent_params if agent_params is not {} else None
    )
    gen_id = new_job_submission.get('_id')
    if gen_id is not None:
        new_job_submission.pop('_id')

    return new_job_submission
    # except Exception as e:
        # raise HTTPException(status_code=404, detail=f"A simularium file cannot be parsed from your input. Please check your input file and refer to the simulariumio documentation for more details.")


@app.post(
    "/get-compatible-for-verification",
    response_model=CompatibleSimulators,
    operation_id='get-compatible-for-verification',
    tags=["Files"],
    summary='Get the simulators that are compatible with either a given OMEX/COMBINE archive or SBML model simulation.')
async def get_compatible_for_verification(
        uploaded_file: UploadFile = File(..., description="Either a COMBINE/OMEX archive or SBML file to be simulated."),
        versions: bool = Query(default=False, description="Whether to include the simulator versions for each compatible simulator."),
) -> CompatibleSimulators:
    try:
        filename = uploaded_file.filename
        simulators = COMPATIBLE_VERIFICATION_SIMULATORS.copy()  # TODO: dynamically extract this!

        # handle filetype: amici is not compatible with sbml verification at the moment
        # if not filename.endswith(".omex"):
        #     for sim in simulators:
        #         if sim[0] == 'amici':
        #             simulators.remove(sim)

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


