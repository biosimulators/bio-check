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
from process_bigraph import ProcessTypes, pp
from pymongo.collection import Collection as MongoCollection

from shared_worker import MongoDbConnector, JobStatus, DatabaseCollections, unique_id, BUCKET_NAME, handle_exception
from log_config import setup_logging
from io_worker import get_sbml_species_mapping, read_h5_reports, download_file, format_smoldyn_configuration, write_uploaded_file
from data_generator import (
    generate_time_course_data,
    generate_composition_result_data,
    run_smoldyn,
    run_readdy,
    handle_sbml_exception,
    generate_biosimulator_utc_outputs,
    generate_sbml_utc_outputs,
    get_output_stack,
    sbml_output_stack
)


# TODO: Create general Worker process implementation!

# for dev only
load_dotenv('../assets/dev/config/.env_dev')


# logging TODO: implement this.
logger = logging.getLogger("biochecknet.job.global.log")
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


_CORE = ProcessTypes()
BIGRAPH_IMPLEMENTATIONS = [
    ('output-generator', 'OutputGenerator'),
    ('time-course-output-generator', 'TimeCourseOutputGenerator'),
    ('smoldyn_step', 'SmoldynStep'),
    ('simularium_smoldyn_step', 'SimulariumSmoldynStep'),
    ('mongo-emitter', 'MongoDatabaseEmitter')
]

APP_PROCESS_REGISTRY, REGISTERED_BIGRAPH_ADDRESSES = register_implementation_addresses(BIGRAPH_IMPLEMENTATIONS, _CORE)


class Supervisor:
    def __init__(self, db_connector: MongoDbConnector, app_process_registry=None, queue_timer: int = 10, preferred_queue_index: int = 0):
        self.db_connector = db_connector
        self.queue_timer = queue_timer
        self.preferred_queue_index = preferred_queue_index
        self.job_queue = self.db_connector.pending_jobs()
        self._supervisor_id: Optional[str] = "supervisor_" + unique_id()
        self.app_process_registry = app_process_registry
        self.logger = logging.getLogger("biochecknet.job.supervisor.log")
        setup_logging(self.logger)

    async def check_jobs(self) -> int:
        """Returns non-zero if max retries reached, zero otherwise.

        # 1. For job (i) in job q, check if jobid exists for any job within db_connector.completed_jobs()
        # 1a. If so, pop the job from the pending queue
        # 2. If job doesnt yet exist in completed, summon a worker.
        # 3. Give the worker the pending job (i)
        # 4. Create completed job in which the job id from # 1 is the job id (id?) and results is worker.job_result
        # 5. Worker automatically is dismissed
        # 5a: TODO: In parallel, keep a pool of n workers List[Worker]. Summon them asynchronously and append more instances as demand increases.
        # 6. Sleep for a larger period of time
        # 7. At the end of check_jobs, run self.job_queue = self.db_connector.pending_jobs() (refresh)
        """
        for _ in range(self.queue_timer):
            # perform check
            await self.run_job_check()
            await sleep(2)

            # refresh jobs
            self.job_queue = self.db_connector.pending_jobs()

        return 0

    async def run_job_check(self):
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
                    # check: verifications
                    elif job_id.startswith('verification'):
                        worker = VerificationWorker(job=pending_job)
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
                    if isinstance(worker, VerificationWorker) or isinstance(worker, CompositionRunWorker):
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
                    error = handle_exception('Job Execution Error')
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

    async def store_registered_addresses(self):
        # store list of process addresses that is available to the client via mongodb:
        confirmation = await self.db_connector.write(
            collection_name="bigraph_registry",
            registered_addresses=REGISTERED_BIGRAPH_ADDRESSES,
            timestamp=self.db_connector.timestamp(),
            version="latest",
            return_document=True
        )
        return confirmation


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
        self.logger = logging.getLogger(f"biochecknet.job.worker-{self.scope}.log")
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

        result = generate_sbml_utc_outputs(sbml_fp=local_fp, start=start, dur=end, steps=steps, simulators=[simulator])
        self.job_result = result[simulator]


class VerificationWorker(Worker):
    def __init__(self, job: Dict, supervisor: Supervisor = None):
        super().__init__(job=job, supervisor=supervisor, scope='verification')
        self.state_result = {}

    async def run(self, selection_list: List[str] = None) -> Dict:
        # process simulation
        input_fp = self.job_params['path']
        selection_list = self.job_params.get('selection_list')
        if input_fp.endswith('.omex'):
            self._execute_omex_job()
        elif input_fp.endswith('.xml'):
            self._execute_sbml_job()

        # select data if applicable
        selections = self.job_params.get("selection_list", selection_list)
        if selections is not None:
            self.job_result = self._select_observables(job_result=self.job_result, observables=selections)

        # calculate rmse
        try:
            rmse_matrix = self._calculate_pairwise_rmse()
            self.job_result['rmse'] = self._format_rmse_matrix(rmse_matrix)
            # self.job_result['rmse'] = self._calculate_pairwise_rmse()
        except:
            e = handle_exception('RMSE Calculation')
            self.logger.error(e)
            self.job_result['rmse'] = {'error': e}

        # simulators = self.job_params.get('simulators')
        # # include expected results in rmse if applicable
        # if self.job_params.get('expected_results') is not None:
        #     simulators.append('expected_results')
        # # calc rmse for each simulator
        # for simulator in simulators:
        #     try:
        #         self.job_result['rmse'][simulator] = self._calculate_inter_simulator_rmse(target_simulator=simulator)
        #     except:
        #         self.job_result['rmse'][simulator] = {}

        return self.job_result

    def _calculate_inter_simulator_rmse(self, target_simulator):
        # extract data fields
        spec_data = self.job_result

        # iterate through observables
        mse_values = []
        for observable, sim_details in spec_data.items():
            mse_data = sim_details['mse'][target_simulator]

            # exclude self-comparison and collect MSE values with other simulators
            for sim, mse in mse_data.items():
                if sim != target_simulator:
                    mse_values.append(mse)

        # calculate the mean of the collected MSE values
        if mse_values:
            mean_mse = sum(mse_values) / len(mse_values)

            # return the square root of the mean MSE (RMSE)
            return math.sqrt(mean_mse)
        # else:
        # handle case where no MSE values are present (to avoid division by zero)
        # return 0.0

    def _format_rmse_matrix(self, matrix) -> dict[str, dict[str, float]]:
        _m = matrix
        rmse = {}

        # iterate over original matrix
        for outer, inner_dict in _m.items():
            keys = list(inner_dict.keys())
            scores = list(inner_dict.values())
            valid_scores = []
            valid_keys = []
            n_valid = 0

            for i, score in enumerate(scores):
                # case: valid score
                if score is not None and not np.isnan(score):
                    valid_keys.append(keys[i])
                    valid_scores.append(score)
                    n_valid += 1

            # dict is valid if greater than 1
            if n_valid > 1:
                inner = dict(zip(valid_keys, valid_scores))
                rmse[outer] = inner

        return rmse

    def _calculate_pairwise_rmse(self) -> dict:
        # get input data
        spec_data = self.job_result
        simulators = self.job_params['simulators']
        if self.job_params.get('expected_results') is not None:
            simulators.append('expected_results')
        n = len(simulators)

        # set up empty matrix
        rmse_matrix = np.zeros((n, n))

        # enumerate over i,j of simulators in a matrix
        for i, sim_i in enumerate(simulators):
            for j, sim_j in enumerate(simulators):
                if i != j:
                    mse_values = []
                    for observable, observable_data in spec_data.items():
                        if not isinstance(observable_data, str):
                            mse_data = observable_data['mse']
                            if sim_j in mse_data:
                                # mse_data[sim_j] is a dict containing MSEs with other simulators
                                for comparison_sim, mse_value in mse_data[sim_j].items():
                                    if comparison_sim == sim_i:
                                        mse_values.append(mse_value)
                    if mse_values:
                        mean_mse = sum(mse_values) / len(mse_values)
                        rmse_matrix[i, j] = math.sqrt(mean_mse)
                    else:
                        # TODO: make this more robust
                        rmse_matrix[i, j] = np.nan
                else:
                    rmse_matrix[i, j] = 0.0

        return pd.DataFrame(rmse_matrix, columns=simulators, index=simulators).to_dict()

    def __calculate_pairwise_rmse(self) -> dict:
        # get input data
        spec_data = self.job_result
        simulators = self.job_params['simulators']
        if self.job_params.get('expected_results') is not None:
            simulators.append('expected_results')
        n = len(simulators)

        # set up empty matrix
        rmse_matrix = np.zeros((n, n))

        # enumerate over i,j of simulators in a matrix
        cols = []
        mse_values = []
        for i, sim_i in enumerate(simulators):
            for j, sim_j in enumerate(simulators):
                if i != j:
                    # fetch mse values
                    for observable, observable_data in spec_data.items():
                        if not isinstance(observable_data, str):
                            mse_data = observable_data['mse']
                            # means that simulator was successful and is in mse matrices
                            if sim_j in mse_data.keys():
                                # append sim_j to cols because it is valid (in the mse matrix)
                                cols.append(sim_j)

                                # mse_data[sim_j] is a dict containing MSEs with other simulators
                                for comparison_sim, mse_value in mse_data[sim_j].items():
                                    if comparison_sim == sim_i:
                                        mse_values.append(mse_value)

                    # # get unique list of valid sim keys and adjust matrix shape accordingly
                    # cols = list(set(cols))
                    # n_valid_simulators = len(cols)
                    # adjusted_matrix_shape = (n_valid_simulators, n_valid_simulators)
                    # rmse_matrix = np.resize(rmse_matrix, adjusted_matrix_shape)

                    # # assign mse_values to newly shaped matrix (should be same shape)
                    # if mse_values:
                    #     mean_mse = sum(mse_values) / len(mse_values)
                    #     rmse_matrix[i, j] = math.sqrt(mean_mse)

                    # else:
                    # TODO: make this more robust
                    # rmse_matrix[i, j] = np.nan
                # else:
                # rmse_matrix[i, j] = 0.0
        # get unique list of valid sim keys and adjust matrix shape accordingly
        columns = list(set(cols))
        n_valid_simulators = len(columns)
        adjusted_matrix_shape = (n_valid_simulators, n_valid_simulators)
        rmse_matrix = np.resize(rmse_matrix, adjusted_matrix_shape)

        # assign mse_values to newly shaped matrix (should be same shape)
        if mse_values:
            for i, sim_i in enumerate(columns):
                for j, sim_output in enumerate(mse_values):
                    if i != j:
                        mse_vals = mse_values[i][j]
                        mean_mse = sum(mse_vals) / len(mse_vals)
                        rmse_matrix[i, j] = math.sqrt(mean_mse)

        return pd.DataFrame(rmse_matrix, columns=columns, index=columns).to_dict()

    def _select_observables(self, job_result, observables: List[str] = None) -> Dict:
        """Select data from the input data that is passed which should be formatted such that the data has mappings of observable names
            to dicts in which the keys are the simulator names and the values are arrays. The data must have content accessible at: `data['content']['results']`.
        """
        outputs = job_result.copy()
        result = {}
        data = job_result

        # case: results from sbml
        if isinstance(data, dict):
            for name, obs_data in data.items():
                if name in observables:
                    result[name] = obs_data
            outputs = result
        # case: results from omex
        elif isinstance(data, list):
            for i, datum in enumerate(data):
                name = datum['species_name']
                if name not in observables:
                    print(f'Name: {name} not in observables')
                    data.pop(i)
            outputs = data

        return outputs

    def _execute_sbml_job(self):
        params = None
        out_dir = tempfile.mkdtemp()
        source_fp = self.job_params['path']
        source_report_fp = self.job_params.get('expected_results')

        # download sbml file
        local_fp = download_file(source_blob_path=source_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

        # get ground truth from bucket if applicable
        local_report_fp = None
        if source_report_fp is not None:
            local_report_fp = download_file(source_blob_path=source_report_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

        simulators = self.job_params.get('simulators', ['amici', 'copasi', 'tellurium'])  # pysces
        include_outs = self.job_params.get('include_outputs', False)
        comparison_id = self.job_params.get('job_id')
        output_start = self.job_params.get('start')
        end = self.job_params.get('end', 10)
        steps = self.job_params.get('steps', 100)
        rtol = self.job_params.get('rTol')
        atol = self.job_params.get('aTol')

        result = self._run_comparison_from_sbml(
            sbml_fp=local_fp,
            start=output_start,
            dur=end,
            steps=steps,
            rTol=rtol,
            aTol=atol,
            ground_truth=local_report_fp,
            simulators=simulators
        )
        self.job_result = result

    def _execute_omex_job(self):
        params = None
        out_dir = tempfile.mkdtemp()
        source_fp = self.job_params['path']
        source_report_fp = self.job_params.get('expected_results')

        # download sbml file
        local_fp = download_file(source_blob_path=source_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

        # get ground truth from bucket if applicable
        truth_vals = None
        local_report_fp = None
        if source_report_fp is not None:
            local_report_fp = download_file(source_blob_path=source_report_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)
            truth_vals = read_h5_reports(local_report_fp)

        simulators = self.job_params.get('simulators', [])
        include_outs = self.job_params.get('include_outputs', False)
        tol = self.job_params.get('rTol')
        atol = self.job_params.get('aTol')
        comparison_id = self.job_params.get('job_id')

        result = self._run_comparison_from_omex(
            path=local_fp,
            simulators=simulators,
            out_dir=out_dir,
            include_outputs=include_outs,
            truth_vals=truth_vals
        )

        self.job_result = result
        # except:
        # error = handle_sbml_exception()
        # logger.error(error)
        # self.job_result = {"error": error}

    def _run_comparison_from_omex(
            self,
            path: str,
            simulators: List[str],
            out_dir: str,
            include_outputs: bool = True,
            truth_vals=None,
            rTol=None,
            aTol=None
    ) -> Dict:
        """Execute a Uniform Time Course comparison for ODE-based simulators from Biosimulators."""
        # download the omex file from GCS
        # source_blob_name = path.replace('gs://bio-check-requests-1', '')  # Assuming omex_fp is the blob name in GCS
        # local_omex_fp = os.path.join(out_dir, path.split('/')[-1])
        # download_blob(bucket_name=BUCKET_NAME, source_blob_name=path, destination_file_name=local_omex_fp)

        # download the report file from GCS if applicable
        # if ground_truth_report_path is not None:
        #     source_report_blob_name = ground_truth_report_path.replace('gs://bio-check-requests-1', '')
        #     # local_report_path = os.path.join(out_dir, source_report_blob_name.split('/')[-1])
        #     local_report_path = os.path.join(out_dir, ground_truth_report_path.split('/')[-1])
        #     truth_vals = read_report_outputs(ground_truth_report_path)
        # else:
        #     truth_vals = None

        results = {}

        # generate the data
        output_data = generate_biosimulator_utc_outputs(omex_fp=path, output_root_dir=out_dir, simulators=simulators, alg_policy="same_framework")
        ground_truth_data = truth_vals.to_dict() if not isinstance(truth_vals, type(None)) else truth_vals

        # generate the species comparisons
        observable_names = []
        for simulator_name in output_data.keys():
            sim_data = output_data[simulator_name]
            if isinstance(sim_data, dict):
                for observable_name in sim_data.keys():
                    observable_names.append(observable_name)
        names = list(set(observable_names))

        for species in names:
            if not species == 'EmptySet':
                # TODO: reimplement this!
                ground_truth_data = None
                # if ground_truth:
                #     for data in ground_truth['data']:
                #         if data['dataset_label'] == species:
                #             ground_truth_data = data['data']
                # generate species comparison

                results[species] = self._generate_omex_utc_species_comparison(
                    output_data=output_data,
                    species_name=species,
                    simulators=simulators,
                    ground_truth=ground_truth_data,
                    rTol=rTol,
                    aTol=aTol
                )

        return results

    def _run_comparison_from_sbml(self, sbml_fp, start, dur, steps, rTol=None, aTol=None, simulators=None, ground_truth=None) -> Dict:
        species_mapping = get_sbml_species_mapping(sbml_fp)
        mapping_names = list(species_mapping.keys())
        data = self._generate_formatted_sbml_outputs(sbml_filepath=sbml_fp, start=start, dur=dur, steps=steps, simulators=simulators)
        self.state_result = data.get('state')
        output_data = data.get('output_data')
        sims = simulators or list(output_data.keys())

        results = {}
        for simulator in sims:
            sim_data = output_data[simulator]
            spec_names = list(sim_data.keys())

            for i, species_name in enumerate(spec_names):
                if not species_name == 'error':
                    species_comparison = self._generate_sbml_utc_species_comparison(
                        output_data=output_data,
                        species_name=species_name,
                        rTol=rTol,
                        aTol=aTol,
                        ground_truth=ground_truth
                    )
                    results[list(species_mapping.keys())[i]] = species_comparison

        return results

    def _generate_omex_utc_species_comparison(self, output_data, species_name, simulators, ground_truth=None, rTol=None, aTol=None):
        # extract valid comparison data
        stack = get_output_stack(outputs=output_data, spec_name=species_name)
        species_comparison_data = {}
        for simulator_name in stack.keys():
            row = stack[simulator_name]
            if row is not None:
                species_comparison_data[simulator_name] = row

        vals = list(species_comparison_data.values())
        valid_sims = list(species_comparison_data.keys())
        outputs = vals
        methods = ['mse', 'proximity']
        if len(outputs) > 1:
            # outputs = _get_output_stack(output_data, species_name)
            matrix_vals = list(map(
                lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=valid_sims, method=m, ground_truth=ground_truth, rtol=rTol, atol=aTol).to_dict(),
                methods
            ))
        else:
            matrix_vals = list(map(
                lambda m: {},
                methods
            ))

        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        data = None
        for simulator_name in output_data.keys():
            sim_output = output_data[simulator_name]
            for spec_name, spec_output in sim_output.items():
                if spec_name == species_name:
                    data = output_data[simulator_name][spec_name]
                elif spec_name.lower() == 'error':
                    data = output_data[simulator_name]

                results['output_data'][simulator_name] = data.tolist() if isinstance(data, np.ndarray) else data
        return results

    def _generate_formatted_sbml_outputs(self, sbml_filepath, start, dur, steps, simulators, ground_truth=None):
        # TODO: stable content is commented-out below. Currently placing alternate function in place of stable content.
        # return generate_sbml_utc_outputs(sbml_fp=sbml_filepath, start=start, dur=dur, steps=steps, simulators=simulators, truth=ground_truth)
        return generate_time_course_data(
            input_fp=sbml_filepath,
            start=start,
            end=dur,
            steps=steps,
            core=APP_PROCESS_REGISTRY,
            simulators=simulators,
            expected_results_fp=ground_truth,
        )

    def _generate_sbml_utc_species_comparison(self, output_data, species_name, ground_truth=None, rTol=None, aTol=None):
        # outputs = sbml_output_stack(spec_name=species_name, output=output_data)
        sims = list(output_data.keys())
        simulators = sims.copy()
        results = {}

        # extract comparison data
        stack = sbml_output_stack(spec_name=species_name, output=output_data)

        species_comparison_data = {}
        for simulator_name in stack.keys():
            row = stack[simulator_name]
            if row is None:
                simulators.remove(simulator_name)
            else:
                species_comparison_data[simulator_name] = row

        vals = list(species_comparison_data.values())
        valid_sims = list(species_comparison_data.keys())
        methods = ['mse', 'proximity']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=vals, simulators=valid_sims, method=m, ground_truth=ground_truth, rtol=rTol, atol=aTol).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))

        # extract output data
        results['output_data'] = {}
        for simulator in sims:
            sim_data = output_data[simulator]
            key = None
            if "error" in sim_data.keys():
                key = 'error'
            else:
                key = species_name
            results['output_data'][simulator] = output_data[simulator].get(key)

        # results['output_data'] = {}
        # for simulator_name in output_data.keys():
        #     for spec_name, output in output_data[simulator_name].items():
        #         data = output_data[simulator_name][spec_name]
        #         results['output_data'][simulator_name] = data.tolist() if isinstance(data, np.ndarray) else data
        #         # if species_name in spec_name:
        #         #     results['output_data'][simulator_name] = data.tolist() if isinstance(data, np.ndarray) else data

        return results

    def _generate_species_comparison_matrix(
            self,
            outputs: Union[np.ndarray, List[np.ndarray]],
            simulators: List[str],
            method: Union[str, any] = 'proximity',
            rtol: float = None,
            atol: float = None,
            ground_truth: np.ndarray = None
    ) -> pd.DataFrame:
        """Generate a Mean Squared Error comparison matrix of arr1 and arr2, indexed by simulators by default,
            or an AllClose Tolerance routine result if `method` is set to `proximity`.

            Args:
                outputs: list of output arrays.
                simulators: list of simulator names.
                method: pass one of either: `mse` to perform a mean-squared error calculation
                    or `proximity` to perform a pair-wise proximity tolerance test using `np.allclose(outputs[i], outputs[i+1])`.
                rtol:`float`: relative tolerance for comparison if `proximity` is used.
                atol:`float`: absolute tolerance for comparison if `proximity` is used.
                ground_truth: If passed, this value is compared against each simulator in simulators. Currently, this
                    field is agnostic to any verified/validated source, and we trust that the user has verified it. Defaults
                    to `None`.

            Returns:
                Pandas dataframe representing a comparison matrix where `i` and `j` are both indexed by the
                    simulators involved. The aforementioned simulators involved will also include the `ground_truth` value
                    within the indices if one is passed.
        """
        # TODO: implement the ground truth
        _simulators = simulators.copy()
        _outputs = outputs.copy()
        if isinstance(_outputs, np.ndarray):
            _outputs = _outputs.tolist()
        if ground_truth is not None:
            _simulators.append('expected_results')
            _outputs.append(ground_truth)
        use_tol_method = method.lower() == 'proximity'
        matrix_dtype = np.float64 if not use_tol_method else bool
        num_simulators = len(_simulators)
        mse_matrix = np.zeros((num_simulators, num_simulators), dtype=matrix_dtype)
        # fill the matrices with the calculated values
        for i in range(len(_simulators)):
            for j in range(i, len(_simulators)):
                try:
                    output_i = _outputs[i]
                    output_j = _outputs[j]
                    method_type = method.lower()
                    if not isinstance(output_i, str) and not isinstance(output_j, str):
                        result = self.calculate_mse(output_i, output_j) if method_type == 'mse' else self.compare_arrays(arr1=output_i, arr2=output_j, rtol=rtol, atol=atol)
                        mse_matrix[i, j] = result
                except ValueError:
                    error = handle_sbml_exception()
                    mse_matrix[i, j] = error
                if i != j:
                    mse_matrix[j, i] = mse_matrix[i, j]
        return pd.DataFrame(mse_matrix, index=_simulators, columns=_simulators)

    def calculate_mse(self, a, b) -> np.float64:
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)
        return np.mean((a - b) ** 2)

    def compare_arrays(self, arr1: np.ndarray, arr2: np.ndarray, atol=None, rtol=None) -> bool:
        """Original methodology copied from biosimulations runutils."""
        max1 = max(arr1)
        max2 = max(arr2)
        aTol = atol or max(1e-3, max1 * 1e-5, max2 * 1e-5)
        rTol = rtol or 1e-4
        return np.allclose(arr1, arr2, rtol=rTol, atol=aTol)


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
    #     from bigraph_steps import generate_simularium_file
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


