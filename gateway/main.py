import json
import os
import logging
import uuid
from typing import *

import dotenv
from tempfile import mkdtemp

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, APIRouter, Body
from fastapi.responses import FileResponse
from pydantic import BeforeValidator
from starlette.middleware.cors import CORSMiddleware

from shared.data_model import (
    ReaddySpeciesConfig,
    ReaddyReactionConfig,
    ReaddyParticleConfig,
    ReaddyRun,
    SmoldynRun,
    DbClientResponse,
    AgentParameters,
    BigraphRegistryAddresses,
    IncompleteJob,
    DB_TYPE,
    DB_NAME,
    BUCKET_NAME,
    JobStatus,
    DatabaseCollections,
    CompositionNode,
    CompositionSpec
)
from shared.database import MongoDbConnector
from shared.io import write_uploaded_file, download_file_from_bucket
from shared.log_config import setup_logging


logger = logging.getLogger("compose.gateway.main.log")
setup_logging(logger)


# -- load dev env -- #

dotenv.load_dotenv("../shared/.env")  # NOTE: create an env config at this filepath if dev


# -- constraints -- #

version_path = os.path.join(
    os.path.dirname(__file__),
    ".VERSION"
)
if os.path.exists(version_path):
    with open(version_path, 'r') as f:
        APP_VERSION = f.read().strip()
else:
    APP_VERSION = "0.0.1"

MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
APP_TITLE = "compose"
APP_ORIGINS = [
    'http://127.0.0.1:8000',
    'http://127.0.0.1:4200',
    'http://127.0.0.1:4201',
    'http://127.0.0.1:4202',
    'http://localhost:4200',
    'http://localhost:4201',
    'http://localhost:4202',
    'http://localhost:8000',
    'http://localhost:3001',
    'https://biosimulators.org',
    'https://www.biosimulators.org',
    'https://biosimulators.dev',
    'https://www.biosimulators.dev',
    'https://run.biosimulations.dev',
    'https://run.biosimulations.org',
    'https://biosimulations.dev',
    'https://biosimulations.org',
    'https://bio.libretexts.org',
    'https://compose.biosimulations.org'
]
APP_SERVERS = [
    # {
    #     "url": "https://compose.biosimulations.org",
    #     "description": "Production server"
    # },
    # {
    #     "url": "http://localhost:3001",
    #     "description": "Main Development server"
    # },
    # {
    #     "url": "http://localhost:8000",
    #     "description": "Alternate Development server"
    # }
]

# -- app components -- #

router = APIRouter()
app = FastAPI(title=APP_TITLE, version=APP_VERSION, servers=APP_SERVERS)

# add origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=APP_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"])

# add servers
# app.servers = APP_SERVERS


# -- mongo db -- #

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
    return {'root': 'https://compose.biosimulations.org'}


# -- submit composition jobs --

@app.post(
    "/submit-composition-spec",
    response_model=CompositionSpec,
)
async def submit_composition_spec(spec_file: UploadFile = File(..., description="Composition JSON File")) -> CompositionSpec:
    # validate filetype
    if not spec_file.filename.endswith('.json') and spec_file.content_type != 'application/json':
        raise HTTPException(status_code=400, detail="Invalid file type. Only JSON files are supported.")

    # parse uploaded file for Composition spec
    try:
        contents = await spec_file.read()
        data: Dict = json.loads(contents)
        nodes = []
        for node_name, node_spec in data.items():
            node = CompositionNode(name=node_name, **node_spec)
            nodes.append(node)

        composition = CompositionSpec(nodes=nodes, emitter_mode="all")

        # write spec to db for job, immediately available to the worker
        await db_connector.write(
            collection_name="jobs",
            status="SUBMITTED:PENDING",
            spec=composition.spec,
        )

        return composition
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# run simulations

@app.post(
    "/run-smoldyn",
    response_model=SmoldynRun,
    name="Run a smoldyn simulation",
    operation_id="run-smoldyn",
    tags=["Simulation Execution"],
    summary="Run a smoldyn simulation.")
async def run_smoldyn(
        uploaded_file: UploadFile = File(..., description="Smoldyn Configuration File"),
        duration: int = Query(default=None, description="Simulation Duration"),
        dt: float = Query(default=None, description="Interval of step with which simulation runs"),
        # initial_molecule_state: List = Body(default=None, description="Mapping of species names to initial molecule conditions including counts and location.")
) -> SmoldynRun:
    try:
        # get job params
        job_id = "simulation-execution-smoldyn" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        uploaded_file_location = await write_uploaded_file(job_id=job_id, uploaded_file=uploaded_file, bucket_name=BUCKET_NAME, extension='.txt')

        # instantiate new return
        smoldyn_run = SmoldynRun(
            job_id=job_id,
            timestamp=_time,
            status=JobStatus.PENDING.value,
            path=uploaded_file_location,
            duration=duration,
            dt=dt,
            simulators=["smoldyn"]
        )

        # insert job
        pending_job = await db_connector.insert_job_async(
            collection_name=DatabaseCollections.PENDING_JOBS.value,
            job_id=smoldyn_run.job_id,
            timestamp=smoldyn_run.timestamp,
            status=smoldyn_run.status,
            path=smoldyn_run.path,
            duration=smoldyn_run.duration,
            dt=smoldyn_run.dt,
            simulators=smoldyn_run.simulators
        )

        # return typed obj
        return smoldyn_run
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/run-readdy",
    response_model=ReaddyRun,
    name="Run a readdy simulation",
    operation_id="run-readdy",
    tags=["Simulation Execution"],
    summary="Run a readdy simulation.")
async def run_readdy(
        box_size: List[float] = Query(default=[0.3, 0.3, 0.3], description="Box Size of box"),
        duration: int = Query(default=10, description="Simulation Duration"),
        dt: float = Query(default=0.0008, description="Interval of step with which simulation runs"),
        species_config: List[ReaddySpeciesConfig] = Body(
            ...,
            description="Species Configuration, specifying species name mapped to diffusion constant",
            examples=[
                [
                    {"name": "E",  "diffusion_constant": 10.0},
                    {"name": "S",  "diffusion_constant": 10.0},
                    {"name": "ES", "diffusion_constant": 10.0},
                    {"name": "P", "diffusion_constant": 10.0}
                ]
            ]
        ),
        reactions_config: List[ReaddyReactionConfig] = Body(
            ...,
            description="Reactions Configuration, specifying reaction scheme mapped to reaction constant.",
            examples=[
                [
                    {"scheme": "fwd: E +(0.03) S -> ES", "rate": 86.78638438},
                    {"scheme": "back: ES -> E +(0.03) S", "rate": 1.0},
                    {"scheme": "prod: ES -> E +(0.03) P", "rate": 1.0},
                ]
            ]
        ),
        particles_config: List[ReaddyParticleConfig] = Body(
            ...,
            description="Particles Configuration, specifying initial particle positions for each particle.",
            examples=[
                [
                    {
                        "name": "E",
                        "initial_positions": [
                            [-0.11010841, 0.01048227, -0.07514985],
                            [0.02715631, -0.03829782, 0.14395517],
                            [0.05522253, -0.11880506, 0.02222362]
                        ]
                    },
                    {
                        "name": "S",
                        "initial_positions": [
                            [-0.21010841, 0.21048227, -0.07514985],
                            [0.02715631, -0.03829782, 0.14395517],
                            [0.05522253, -0.11880506, 0.02222362]
                        ]
                    }
                ]
            ]
        ),
        unit_system_config: Dict[str, str] = Body({"length_unit": "micrometer", "time_unit": "second"}, description="Unit system configuration"),
        reaction_handler: str = Query(default="UncontrolledApproximation", description="Reaction handler as per Readdy simulation documentation.")
) -> ReaddyRun:
    try:
        # get job params
        job_id = "simulation-execution-readdy" + str(uuid.uuid4())
        _time = db_connector.timestamp()

        # instantiate new return
        readdy_run = ReaddyRun(
            job_id=job_id,
            timestamp=_time,
            box_size=box_size,
            status=JobStatus.PENDING.value,
            duration=duration,
            dt=dt,
            simulators=["readdy"],
            species_config=species_config,
            reactions_config=reactions_config,
            particles_config=particles_config,
            unit_system_config=unit_system_config,
            reaction_handler=reaction_handler,
        )

        # insert job
        pending_job = await db_connector.insert_job_async(
            collection_name=DatabaseCollections.PENDING_JOBS.value,
            box_size=readdy_run.box_size,
            job_id=readdy_run.job_id,
            timestamp=readdy_run.timestamp,
            status=readdy_run.status,
            duration=readdy_run.duration,
            dt=readdy_run.dt,
            simulators=readdy_run.simulators,
            species_config=[config.model_dump() for config in readdy_run.species_config],
            reactions_config=[config.model_dump() for config in readdy_run.reactions_config],
            particles_config=[config.model_dump() for config in readdy_run.particles_config],
            unit_system_config=readdy_run.unit_system_config,
            reaction_handler=readdy_run.reaction_handler
        )

        # return typed obj
        return readdy_run
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


# TODO: create an implementation of the bigraph node spec via data model here instead of arbitrary dict
@app.post(
    "/run-composition",
    name="Run a process bigraph composition",
    operation_id="run-composition",
    tags=["Composition"],
    summary="Run a process bigraph composition.")
async def run_composition(
        spec: Dict[str, Any] = Body(..., description="Process bigraph specification"),
):
    try:
        job_id = "composition" + str(uuid.uuid4())
        _time = db_connector.timestamp()
        pending_composition_job = {
            'status': JobStatus.PENDING.value,
            'state_spec': spec,
            'timestamp': _time,
        }
        await db_connector.write(
            collection_name=DatabaseCollections.PENDING_JOBS.value,
            **pending_composition_job,
        )

        return pending_composition_job
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/get-output-file/{job_id}",
    operation_id='get-output-file',
    tags=["Results"],
    summary='Get the results of an existing simulation run from Smoldyn or Readdy as either a downloadable file or job progression status.'
)
async def get_output_file(job_id: str):
    # state-case: job is completed
    if not job_id.startswith("simulation-execution"):
        raise HTTPException(status_code=404, detail="This must be an output file job query starting with 'simulation-execution'.")

    job = await db_connector.read(collection_name="completed_jobs", job_id=job_id)
    if job is not None:
        # rm mongo index
        job.pop('_id', None)

        # parse filepath in bucket and create file response
        job_data = job
        if isinstance(job_data, dict):
            remote_fp = job_data.get("results").get("results_file")
            if remote_fp is not None:
                temp_dest = mkdtemp()
                local_fp = download_file_from_bucket(source_blob_path=remote_fp, out_dir=temp_dest, bucket_name=BUCKET_NAME)

                # return downloadable file blob
                return FileResponse(path=local_fp, media_type="application/octet-stream", filename=local_fp.split("/")[-1])  # TODO: return special smoldyn file instance

    # state-case: job has failed
    if job is None:
        job = await db_connector.read(collection_name="failed_jobs", job_id=job_id)

    # state-case: job is not in completed:
    if job is None:
        job = await db_connector.read(collection_name="in_progress_jobs", job_id=job_id)

    # state-case: job is not in progress:
    if job is None:
        job = await db_connector.read(collection_name="pending_jobs", job_id=job_id)

    # case: job is either failed, in prog, or pending
    if job is not None:
        # rm mongo index
        job.pop('_id', None)

        # specify source safely
        src = job.get('source', job.get('path'))
        if src is not None:
            source = src.split('/')[-1]
        else:
            source = None

        # return json job status
        return IncompleteJob(
            job_id=job_id,
            timestamp=job.get('timestamp'),
            status=job.get('status'),
            source=source
        )


# BELOW IS THE EXISTING GET OUTPUT!
@app.get(
    "/get-output/{job_id}",
    # response_model=OutputData,
    operation_id='get-output',
    tags=["Results"],
    summary='Get the results of an existing simulation run.')
async def fetch_results(job_id: str):
    # TODO: refactor this!

    # state-case: job is completed
    job = await db_connector.read(collection_name="completed_jobs", job_id=job_id)
    if job is not None:
        job.pop('_id', None)
        job_data = job

        # output-case: output content in dict is a downloadable file
        if isinstance(job_data, dict):
            remote_fp = job_data.get("results").get("results_file")
            if remote_fp is not None:
                temp_dest = mkdtemp()
                local_fp = download_file_from_bucket(source_blob_path=remote_fp, out_dir=temp_dest, bucket_name=BUCKET_NAME)

                return FileResponse(path=local_fp, media_type="application/octet-stream", filename=local_fp.split("/")[-1])
                # return {'path': local_fp, 'media_type': 'application/octet-stream', 'filename': local_fp.split('/')[-1]}

        # otherwise, return job content
        return {'content': job}

    # state-case: job has failed
    if job is None:
        job = await db_connector.read(collection_name="failed_jobs", job_id=job_id)
        if job is not None:
            job.pop('_id', None)
            return {'content': job}

    # state-case: job is not in completed:
    if job is None:
        job = await db_connector.read(collection_name="in_progress_jobs", job_id=job_id)

    # state-case: job is not in progress:
    if job is None:
        job = await db_connector.read(collection_name="pending_jobs", job_id=job_id)

    # return-case: job exists as either completed, failed, in_progress, or pending
    if not isinstance(job, type(None)):
        # remove autogen obj
        job.pop('_id', None)

        # status/content-case: case: job is completed
        if job['status'] == "COMPLETED":
            # check output for type (either raw data or file download)
            job_data = None
            results = job['results']
            if isinstance(results, dict):
                job_data = job['results'].get('results')
            else:
                job_data = job['results']

            # job has results
            if job_data is not None:
                remote_fp = None

                if isinstance(job_data, list):
                    # return OutputData(content=job)
                    return {'content': job}

                # output-type-case: output is saved as a dict
                if isinstance(job_data, dict):
                    # output-case: output content in dict is a downloadable file
                    if "results_file" in job_data.keys():
                        remote_fp = job_data['results_file']
                    # status/output-case: job is complete and output content is raw data and so return the full data TODO: do something better here
                    else:
                        # return OutputData(content=job)
                        return {'content': job}

                # output-type-case: output is saved as flattened (str) and thus also a file download
                elif isinstance(job_data, str):
                    remote_fp = job_data

                # content-case: output content relates to file download
                if remote_fp is not None:
                    temp_dest = mkdtemp()
                    local_fp = download_file_from_bucket(source_blob_path=remote_fp, out_dir=temp_dest, bucket_name=BUCKET_NAME)

                    # return FileResponse(path=local_fp, media_type="application/octet-stream", filename=local_fp.split("/")[-1])
                    return {'path': local_fp, 'media_type': 'application/octet-stream', 'filename': local_fp.split('/')[-1]}

        # status/content-case: job is either pending or in progress and does not contain files to download
        else:
            # acknowledge the user submission to differentiate between original submission
            status = job['status']
            job['status'] = 'SUBMITTED:' + status

            # return OutputData(content=job)
            return {'content': job}

    # return-case: no job exists in any collection by that id
    else:
        msg = f"Job with id: {job_id} not found. Please check the job_id and try again."
        logger.error(msg)
        raise HTTPException(status_code=404, detail=msg)


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
    # upload_prefix, bucket_prefix = file_upload_prefix(job_id)
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





# @app.get(
#     "/get-process-bigraph-addresses",
#     operation_id="get-process-bigraph-addresses",
#     response_model=BigraphRegistryAddresses,
#     tags=["Composition"],
#     summary="Get process bigraph implementation addresses for composition specifications.")
# async def get_process_bigraph_addresses() -> BigraphRegistryAddresses:
#     registry = await db_connector.read(collection_name="bigraph_registry", version="latest")
#     if registry is not None:
#         addresses = registry.get('registered_addresses')
#         version = registry.get('version')
#
#         return BigraphRegistryAddresses(registered_addresses=addresses, version=version)
#     else:
#         raise HTTPException(status_code=500, detail="Addresses not found.")


@app.get(
    "/get-process-bigraph-addresses",
    operation_id="get-process-bigraph-addresses",
    response_model=BigraphRegistryAddresses,
    tags=["Composition"],
    summary="Get process bigraph implementation addresses for composition specifications.")
async def get_process_bigraph_addresses() -> BigraphRegistryAddresses:
    from bsp import app_registrar
    addresses = list(app_registrar.core.process_registry.registry.keys())
    version = "latest"
    return BigraphRegistryAddresses(registered_addresses=addresses, version=version)
    # else:
    #     raise HTTPException(status_code=500, detail="Addresses not found.")


@app.get(
    "/get-composition-state/{job_id}",
    operation_id="get-composition-state",
    tags=["Composition"],
    summary="Get the composite spec of a given simulation run indexed by job_id.")
async def get_composition_state(job_id: str):
    try:
        spec = await db_connector.read(collection_name="result_states", job_id=job_id)
        if "_id" in spec.keys():
            spec.pop("_id")

        return spec
    except Exception as e:
        logger.error(str(e))
        raise HTTPException(status_code=500, detail=f"No specification found for job with id: {job_id}.")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


