import logging
import math
import os
import tempfile
from abc import ABC, abstractmethod
from asyncio import sleep
from typing import *

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from process_bigraph import ProcessTypes
from pymongo.collection import Collection as MongoCollection
from bsp.data_generators import (
    generate_time_course_data,
    generate_composition_result_data,
    run_smoldyn,
    run_readdy,
    handle_sbml_exception,
    generate_sbml_outputs
)
from bsp import app_registrar

from shared.data_model import JobStatus, DatabaseCollections, BUCKET_NAME
from shared.database import MongoDbConnector
from shared.utils import unique_id, handle_sbml_exception
from shared.log_config import setup_logging
from shared.io import get_sbml_species_mapping, download_file, format_smoldyn_configuration, write_uploaded_file



# TODO: Create general Worker process implementation!

# for dev only
load_dotenv('../assets/dev/config/.env_dev')


# logging TODO: implement this.
logger = logging.getLogger("compose.job.global.log")
setup_logging(logger)


def register_implementation_addresses(
        implementations: List[Tuple[str, str]],
        core_registry: ProcessTypes
) -> Tuple[ProcessTypes, List[str]]:
    for process_name, class_name in implementations:
        try:
            import_statement = f'data_generator'
            module = __import__(import_statement)
            bigraph_class = getattr(module, class_name)
            # Register the process
            core_registry.process_registry.register(process_name, bigraph_class)
        except Exception as e:
            logger.warning(f"Cannot register {class_name}. Error:\n**\n{e}\n**")
            continue

    return core_registry, list(core_registry.process_registry.registry.keys())


# _CORE = ProcessTypes()
# BIGRAPH_IMPLEMENTATIONS = [
#     ('output-generator', 'OutputGenerator'),
#     ('time-course-output-generator', 'TimeCourseOutputGenerator'),
#     ('smoldyn_step', 'SmoldynStep'),
#     ('simularium_smoldyn_step', 'SimulariumSmoldynStep'),
#     ('mongo-emitter', 'MongoDatabaseEmitter')
# ]
# APP_PROCESS_REGISTRY, REGISTERED_BIGRAPH_ADDRESSES = register_implementation_addresses(BIGRAPH_IMPLEMENTATIONS, _CORE)


class Supervisor:
    def __init__(self, db_connector: MongoDbConnector, app_process_registry=None, queue_timer: int = 10, preferred_queue_index: int = 0):
        self.db_connector = db_connector
        self.queue_timer = queue_timer
        self.preferred_queue_index = preferred_queue_index
        self.job_queue = self.db_connector.pending_jobs()
        self._supervisor_id: Optional[str] = "supervisor_" + unique_id()
        self.app_process_registry = app_process_registry
        self.logger = logging.getLogger("compose.job.supervisor.log")
        setup_logging(self.logger)

    async def check_jobs(self) -> int:
        for _ in range(self.queue_timer):
            # perform check
            await self._run_job_check()
            await sleep(2)

            # refresh jobs
            self.job_queue = self.db_connector.pending_jobs()

        return 0

    async def _run_job_check(self):
        worker = None
        for i, pending_job in enumerate(self.job_queue):
            # get job params
            job_id = pending_job.get('job_id')
            source = pending_job.get('path')
            source_name = source.split('/')[-1] if source is not None else "No-Source-File"

            # check terminal collections for job
            job_completed = self.job_exists(job_id=job_id, collection_name="completed_jobs")
            job_failed = self.job_exists(job_id=job_id, collection_name="failed_jobs")

            # case: job is not complete, otherwise do nothing
            if not job_completed and not job_failed:
                # change job status for client by inserting a new in progress job
                job_in_progress = self.job_exists(job_id=job_id, collection_name="in_progress_jobs")
                if not job_in_progress:
                    in_progress_entry = {
                        'job_id': job_id,
                        'timestamp': self.db_connector.timestamp(),
                        'status': JobStatus.IN_PROGRESS.value,
                        'requested_simulators': pending_job.get('simulators'),
                        'source': source
                    }

                    # special handling of composition jobs TODO: move this to the supervisor below
                    if job_id.startswith('composition-run'):
                        in_progress_entry['composite_spec'] = pending_job.get('composite_spec')
                        in_progress_entry['simulator'] = pending_job.get('simulators')
                        in_progress_entry['duration'] = pending_job.get('duration')

                    # insert new inprogress job with the same job_id
                    in_progress_job = await self.db_connector.insert_job_async(
                        collection_name="in_progress_jobs",
                        **in_progress_entry
                    )

                    # remove job from pending
                    self.db_connector.db.pending_jobs.delete_one({'job_id': job_id})

                # run job again
                try:
                    # check: run simulations
                    if job_id.startswith('simulation-execution'):
                        worker = SimulationRunWorker(job=pending_job)
                    # check: files
                    elif job_id.startswith('files'):
                        worker = FilesWorker(job=pending_job)
                    elif job_id.startswith('composition'):
                        worker = CompositionRunWorker(job=pending_job)

                    # when worker completes, dismiss worker (if in parallel)
                    await worker.run()

                    # create new completed job using the worker's job_result
                    result_data = worker.job_result
                    await self.db_connector.write(
                        collection_name=DatabaseCollections.COMPLETED_JOBS.value,
                        job_id=job_id,
                        timestamp=self.db_connector.timestamp(),
                        status=JobStatus.COMPLETED.value,
                        results=result_data,
                        source=source_name,
                        requested_simulators=pending_job.get('simulators')
                    )

                    # store the state result if composite (currently only verification and Composition)
                    if isinstance(worker, CompositionRunWorker):
                        state_result = worker.state_result
                        await self.db_connector.write(
                            collection_name="result_states",
                            job_id=job_id,
                            timestamp=self.db_connector.timestamp(),
                            source=source_name,
                            state=state_result,
                        )

                    # remove in progress job
                    self.db_connector.db.in_progress_jobs.delete_one({'job_id': job_id})
                except:
                    # save new execution error to db
                    error = handle_sbml_exception()
                    self.logger.error(error)
                    await self.db_connector.write(
                        collection_name="failed_jobs",
                        job_id=job_id,
                        timestamp=self.db_connector.timestamp(),
                        status=JobStatus.FAILED.value,
                        results=error,
                        source=source_name
                    )
                    # remove in progress job TODO: refactor this
                    self.db_connector.db.in_progress_jobs.delete_one({'job_id': job_id})

    def job_exists(self, job_id: str, collection_name: str) -> bool:
        """Returns True if job with the given job_id exists, False otherwise."""
        unique_id_query = {'job_id': job_id}
        coll: MongoCollection = self.db_connector.db[collection_name]
        job = coll.find_one(unique_id_query) or None

        return job is not None

    async def store_registered_addresses(self, core: ProcessTypes):
        # store list of process addresses whose origin currently is Biosimulator Processes TODO: make this more general
        registered_processes = list(core.process_registry.registry.keys())
        stamp = self.db_connector.timestamp()
        await self.db_connector.write(
            collection_name="processes",
            registered_addresses=registered_processes,
            timestamp=stamp,
            version=f"latest_{stamp}",
            return_document=True
        )

        # store types? TODO: do this in biosimulator processes


# run singularity in docker 1 batch mode 1 web version
class Worker(ABC):
    job_params: Dict
    job_id: str
    job_result: Dict | None
    job_failed: bool
    supervisor: Supervisor
    logger: logging.Logger

    def __init__(self, job: Dict, scope: str, supervisor: Supervisor = None):
        """
        Args:
            job: job parameters received from the supervisor (who gets it from the db) which is a document from the pending_jobs collection within mongo.
        """
        self.job_params = job
        self.job_id = self.job_params['job_id']
        self.job_result = {}
        self.job_failed = False

        # for parallel processing in a pool of workers. TODO: eventually implement this.
        self.worker_id = unique_id()
        self.supervisor = supervisor
        self.scope = scope
        self.logger = logging.getLogger(f"compose.job.worker-{self.scope}.log")
        setup_logging(self.logger)

    @abstractmethod
    async def run(self):
        pass

    def result(self) -> tuple[dict, bool]:
        return (self.job_result, self.job_failed)


class CompositionRunWorker(Worker):
    def __init__(self, job: Dict, supervisor: Supervisor = None):
        super().__init__(job=job, supervisor=supervisor, scope='composition')
        self.state_result = {}

    async def run(self):
        spec = self.job_params.get('state_spec')
        duration = self.job_params.get('duration')
        await self.generate_composition_result_data(state_spec=spec, duration=duration)
        return self.job_result

    async def generate_composition_result_data(self, state_spec, duration):
        result = generate_composition_result_data(state_spec=state_spec, core=APP_PROCESS_REGISTRY, duration=duration)
        self.job_result = {'results': result['results']}
        self.state_result = result.get('state_result')


class SimulationRunWorker(Worker):
    def __init__(self, job: Dict):
        super().__init__(job=job, scope='simulation-run')

    async def run(self):
        # check which endpoint methodology to implement
        source_fp = self.job_params.get('path')
        if source_fp is not None:
            # case: job requires some sort of input file and is thus either a smoldyn, utc, or verification run
            out_dir = tempfile.mkdtemp()
            local_fp = download_file(source_blob_path=source_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

            # case: is a smoldyn job
            if local_fp.endswith('.txt'):
                await self.run_smoldyn(local_fp)
            # case: is utc job
            elif local_fp.endswith('.xml'):
                await self.run_utc(local_fp)
        elif "readdy" in self.job_id:
            # case: job is configured manually by user request
            await self.run_readdy()

        return self.job_result

    async def run_smoldyn(self, local_fp: str):
        # format model file for disabling graphics
        format_smoldyn_configuration(filename=local_fp)

        # get job params
        duration = self.job_params.get('duration')
        dt = self.job_params.get('dt')
        initial_species_state = self.job_params.get('initial_molecule_state')  # not yet implemented

        # execute simularium, pointing to a filepath that is returned by the run smoldyn call
        result = run_smoldyn(model_fp=local_fp, duration=duration, dt=dt)

        # TODO: Instead use the composition framework to do this

        # write the aforementioned output file (which is itself locally written to the temp out_dir, to the bucket if applicable
        results_file = result.get('results_file')
        if results_file is not None:
            uploaded_file_location = await write_uploaded_file(job_id=self.job_id, uploaded_file=results_file, bucket_name=BUCKET_NAME, extension='.txt')
            self.job_result = {'results_file': uploaded_file_location}
        else:
            self.job_result = result

    async def run_readdy(self):
        # get request params
        duration = self.job_params.get('duration')
        dt = self.job_params.get('dt')
        box_size = self.job_params.get('box_size')
        species_config = self.job_params.get('species_config')
        particles_config = self.job_params.get('particles_config')
        reactions_config = self.job_params.get('reactions_config')
        unit_system_config = self.job_params.get('unit_system_config')

        # run simulations
        result = run_readdy(
            box_size=box_size,
            species_config=species_config,
            particles_config=particles_config,
            reactions_config=reactions_config,
            unit_system_config=unit_system_config,
            duration=duration,
            dt=dt
        )

        # extract results file and write to bucket
        results_file = result.get('results_file')
        if results_file is not None:
            uploaded_file_location = await write_uploaded_file(job_id=self.job_id, uploaded_file=results_file, bucket_name=BUCKET_NAME, extension='.h5')
            self.job_result = {'results_file': uploaded_file_location}
            os.remove(results_file)
        else:
            self.job_result = result

    async def run_utc(self, local_fp: str):
        start = self.job_params['start']
        end = self.job_params['end']
        steps = self.job_params['steps']
        simulator = self.job_params.get('simulators')[0]

        result = generate_sbml_outputs(sbml_fp=local_fp, start=start, dur=end, steps=steps, simulators=[simulator])
        self.job_result = result[simulator]


class FilesWorker(Worker):
    def __init__(self, job):
        super().__init__(job, scope='files')

    async def run(self):
        job_id = self.job_params['job_id']
        input_path = self.job_params.get('path')

        try:
            # is a job related to a client file upload
            if input_path is not None:
                # download the input file
                dest = tempfile.mkdtemp()
                local_input_path = download_file(source_blob_path=input_path, bucket_name=BUCKET_NAME, out_dir=dest)

                # case: is a smoldyn output file and thus a simularium job
                # if local_input_path.endswith('.txt'):
                #     await self._run_simularium(job_id=job_id, input_path=local_input_path, dest=dest)
        except Exception as e:
            self.job_result = {'results': str(e)}

        return self.job_result

    # async def _run_simularium(self, job_id: str, input_path: str, dest: str):
    #     from main.py import generate_simularium_file
    #     # get parameters from job
    #     box_size = self.job_params['box_size']
    #     translate = self.job_params['translate_output']
    #     validate = self.job_params['validate_output']
    #     params = self.job_params.get('agent_parameters')
    #     # generate file
    #     result = generate_simularium_file(
    #         input_fp=input_path,
    #         dest_dir=dest,
    #         box_size=box_size,
    #         translate_output=translate,
    #         run_validation=validate,
    #         agent_parameters=params
    #     )
    #     # upload file to bucket
    #     results_file = result.get('simularium_file')
    #     uploaded_file_location = None
    #     if results_file is not None:
    #         if not results_file.endswith('.simularium'):
    #             results_file += '.simularium'
    #         uploaded_file_location = await write_uploaded_file(job_id=job_id, bucket_name=BUCKET_NAME, uploaded_file=results_file, extension='.simularium')
    #     # set uploaded file as result
    #     self.job_result = {'results_file': uploaded_file_location}


