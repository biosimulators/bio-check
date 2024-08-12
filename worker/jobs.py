# -- This should serve as the main file for worker container -- #
import math
import os
import tempfile
import uuid
import logging
from asyncio import sleep
from dataclasses import dataclass
from functools import partial
from typing import *
from dotenv import load_dotenv
from pymongo.collection import Collection as MongoCollection

import numpy as np
import pandas as pd

from shared import BaseClass, MongoDbConnector, download_blob, setup_logging
from data_model import UtcComparison, SimulationError, UtcSpeciesComparison
from io_worker import get_sbml_species_names, get_sbml_model_file_from_archive, read_report_outputs
from output_data import generate_biosimulator_utc_outputs, _get_output_stack, sbml_output_stack, generate_sbml_utc_outputs, get_sbml_species_mapping


# for dev only
load_dotenv('../assets/.env_dev')

# logging
LOGFILE = "biochecknet_worker_jobs.log"
logger = logging.getLogger(__name__)
setup_logging(LOGFILE)

# constraints
DB_TYPE = "mongo"  # ie: postgres, etc
DB_NAME = "service_requests"
BUCKET_NAME = os.getenv("BUCKET_NAME")


def unique_id():
    return str(uuid.uuid4())


async def load_arrows(timer):
    check_timer = timer
    ell = ""
    bars = ""
    msg = "|"
    n_ellipses = timer
    log_interval = check_timer / n_ellipses
    for n in range(n_ellipses):
        single_interval = log_interval / 3
        await sleep(single_interval)
        bars += "="
        disp = bars + ">"
        if n == n_ellipses - 1:
            disp += "|"
        print(disp)


# -- WORKER: "Toolkit" => Has all of the tooling necessary to process jobs.


class Worker:
    def __init__(self, job: Dict):
        self.job_params = job
        self.job_result = None

        # for parallel processing in a pool of workers. TODO: eventually implement this.
        # self.worker_id = unique_id()

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

        # calculate rmse for each simulator over all observables
        self.job_result['rmse'] = {}
        simulators = self.job_params.get('simulators')
        if self.job_params.get('expected_results') is not None:
            simulators.append('ground_truth')
        for simulator in simulators:
            self.job_result['rmse'][simulator] = self._calculate_inter_simulator_rmse(target_simulator=simulator)

        return self.job_result

    def _calculate_inter_simulator_rmse(self, target_simulator):
        # extract data fields
        try:
            spec_data = self.job_result['results']
            spec_names = list(spec_data.keys())
            num_species = len(spec_names)

            squared_differences = []
            # iterate through observables
            for observable, sim_details in spec_data.items():
                mse_data = sim_details['mse'][target_simulator]

                # exclude the self-comparison, calculate squared differences with others
                for sim, mse in mse_data.items():
                    if sim != target_simulator:
                        squared_differences.append(mse ** 2)

            # calc mean of squared diffs
            mean_squared_diff = sum(squared_differences) / len(squared_differences)

            # return the square root of the mean of squared diffs
            return math.sqrt(mean_squared_diff)
        except Exception as e:
            logger.error(msg=e)
            return None

    def _select_observables(self, job_result, observables: List[str] = None) -> Dict:
        """Select data from the input data that is passed which should be formatted such that the data has mappings of observable names
            to dicts in which the keys are the simulator names and the values are arrays. The data must have content accessible at: `data['content']['results']`.
        """
        outputs = job_result.copy()
        result = {}
        data = job_result['results']

        # case: results from sbml
        if isinstance(data, dict):
            for name, obs_data in data.items():
                if name in observables:
                    result[name] = obs_data
            outputs['results'] = result
        # case: results from omex
        elif isinstance(data, list):
            for i, datum in enumerate(data):
                name = datum['species_name']
                if name not in observables:
                    print(f'Name: {name} not in observables')
                    data.pop(i)
            outputs['results'] = data

        return outputs

    def _execute_sbml_job(self):
        params = None
        out_dir = tempfile.mkdtemp()

        # download sbml file
        source_sbml_blob_name = self.job_params['path']
        local_fp = os.path.join(out_dir, source_sbml_blob_name.split('/')[-1])
        download_blob(bucket_name=BUCKET_NAME, source_blob_name=source_sbml_blob_name, destination_file_name=local_fp)

        # get ground truth from bucket if applicable
        ground_truth_report_path = self.job_params.get('expected_results')
        truth_vals = None
        if ground_truth_report_path is not None:
            source_report_blob_name = self.job_params['expected_results']
            local_report_path = os.path.join(out_dir, ground_truth_report_path.split('/')[-1])
            download_blob(bucket_name=BUCKET_NAME, source_blob_name=source_report_blob_name, destination_file_name=local_report_path)
            ground_truth_report_path = local_report_path

        try:
            simulators = self.job_params.get('simulators', [])
            include_outs = self.job_params.get('include_outputs', False)
            comparison_id = self.job_params.get('job_id')
            output_start = self.job_params.get('start')
            end = self.job_params.get('end', 10)
            steps = self.job_params.get('steps', 100)
            rtol = self.job_params.get('rTol')
            atol = self.job_params.get('aTol')

            result = self._run_comparison_from_sbml(sbml_fp=local_fp, start=output_start, dur=end, steps=steps, rTol=rtol, aTol=atol, ground_truth=ground_truth_report_path)
            self.job_result = result
        except Exception as e:
            self.job_result = {"bio-check-message": f"Job for {self.job_params['comparison_id']} could not be completed because:\n{str(e)}"}

    def _execute_omex_job(self):
        params = None
        out_dir = tempfile.mkdtemp()

        # get omex from bucket
        source_omex_blob_name = self.job_params['path']
        local_omex_fp = os.path.join(out_dir, source_omex_blob_name.split('/')[-1])
        download_blob(bucket_name=BUCKET_NAME, source_blob_name=source_omex_blob_name, destination_file_name=local_omex_fp)

        # get ground truth from bucket if applicable
        ground_truth_report_path = self.job_params['expected_results']
        truth_vals = None
        if ground_truth_report_path is not None:
            source_report_blob_name = self.job_params['expected_results']
            local_report_path = os.path.join(out_dir, ground_truth_report_path.split('/')[-1])
            download_blob(bucket_name=BUCKET_NAME, source_blob_name=source_report_blob_name, destination_file_name=local_report_path)
            truth_vals = read_report_outputs(local_report_path)

        try:
            simulators = self.job_params.get('simulators', [])
            include_outs = self.job_params.get('include_outputs', False)
            tol = self.job_params.get('rTol')
            atol = self.job_params.get('aTol')
            comparison_id = self.job_params.get('job_id')

            result = self._run_comparison_from_omex(
                path=local_omex_fp,
                simulators=simulators,
                out_dir=out_dir,
                include_outputs=include_outs,
                comparison_id=comparison_id,
                truth_vals=truth_vals
            )
            self.job_result = result
        except Exception as e:
            self.job_result = {"bio-check-message": f"Job for {self.job_params['job_id']} could not be completed because:\n{str(e)}"}

    def _run_comparison_from_omex(
            self,
            path: str,
            simulators: List[str],
            out_dir: str,
            include_outputs: bool = True,
            comparison_id: str = None,
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

        # run comparison
        comparison_id = comparison_id or 'biosimulators-utc-comparison'
        ground_truth_data = truth_vals.to_dict() if not isinstance(truth_vals, type(None)) else truth_vals

        comparison = self._generate_omex_utc_comparison(
            omex_fp=path,  # path,
            out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
            simulators=simulators,
            comparison_id=comparison_id,
            ground_truth=ground_truth_data,
            rTol=rTol,
            aTol=aTol
        )
        return comparison
        # parse data for return vals
        # spec_comparisons = []
        # for spec_name, comparison_data in comparison['results'].items():
        #     species_comparison = UtcSpeciesComparison(
        #         mse=comparison_data['mse'],
        #         proximity=comparison_data['proximity'],
        #         output_data=comparison_data.get('output_data') if include_outputs else {},
        #         species_name=spec_name)
        #     spec_comparisons.append(species_comparison)
#
        # return UtcComparison(
        #     results=spec_comparisons,
        #     id=comparison_id,
        #     simulators=simulators)

    def _run_comparison_from_sbml(self, sbml_fp, start, dur, steps, rTol=None, aTol=None, simulators=None, ground_truth=None) -> Dict:
        species_mapping = get_sbml_species_mapping(sbml_fp)
        results = {'results': {}}
        for species_name in species_mapping.keys():
            species_comparison = self._generate_sbml_utc_species_comparison(
                sbml_filepath=sbml_fp,
                start=start,
                dur=dur,
                steps=steps,
                species_name=species_name,
                rTol=rTol,
                aTol=aTol,
                simulators=simulators,
                ground_truth=ground_truth
            )
            results['results'][species_name] = species_comparison

        return results

    def _generate_omex_utc_comparison(self, omex_fp, out_dir, simulators, comparison_id, ground_truth=None, rTol=None, aTol=None):
        model_file = get_sbml_model_file_from_archive(omex_fp, out_dir)
        sbml_species_names = get_sbml_species_names(model_file)
        results = {'results': {}, 'comparison_id': comparison_id}
        for i, species in enumerate(sbml_species_names):
            ground_truth_data = None
            if ground_truth:
                for data in ground_truth['data']:
                    if data['dataset_label'] == species:
                        ground_truth_data = data['data']
            results['results'][species] = self._generate_omex_utc_species_comparison(
                omex_fp=omex_fp,
                out_dir=out_dir,
                species_name=species,
                simulators=simulators,
                ground_truth=ground_truth_data,
                rTol=rTol,
                aTol=aTol
            )
        return results

    def _generate_omex_utc_species_comparison(self, omex_fp, out_dir, species_name, simulators, ground_truth=None, rTol=None, aTol=None):
        output_data = generate_biosimulator_utc_outputs(omex_fp, out_dir, simulators)
        outputs = _get_output_stack(output_data, species_name)
        methods = ['mse', 'proximity']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth, rtol=rTol, atol=aTol).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        for simulator_name in output_data.keys():
            for output in output_data[simulator_name]['data']:
                if output['dataset_label'] in species_name:
                    results['output_data'][simulator_name] = output['data'].tolist()
        return results

    def _generate_sbml_utc_species_comparison(self, sbml_filepath, start, dur, steps, species_name, simulators=None, ground_truth=None, rTol=None, aTol=None):
        simulators = simulators or ['copasi', 'tellurium']
        if "amici" in simulators:
            simulators.remove("amici")

        output_data = generate_sbml_utc_outputs(sbml_fp=sbml_filepath, start=start, dur=dur, steps=steps, truth=ground_truth)
        outputs = sbml_output_stack(species_name, output_data)
        methods = ['mse', 'proximity']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth, rtol=rTol, atol=aTol).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        for simulator_name in output_data.keys():
            for spec_name, output in output_data[simulator_name].items():
                if species_name in spec_name:
                    results['output_data'][simulator_name] = output_data[simulator_name][spec_name].tolist()
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
            _simulators.append('ground_truth')
            _outputs.append(ground_truth)

        use_tol_method = method.lower() == 'proximity'
        matrix_dtype = np.float64 if not use_tol_method else bool
        num_simulators = len(_simulators)
        mse_matrix = np.zeros((num_simulators, num_simulators), dtype=matrix_dtype)

        # fill the matrices with the calculated values
        for i in range(len(_simulators)):
            for j in range(i, len(_simulators)):
                output_i = _outputs[i]
                output_j = _outputs[j]
                method_type = method.lower()
                result = self.calculate_mse(output_i, output_j) if method_type == 'mse' else self.compare_arrays(arr1=output_i, arr2=output_j, rtol=rtol, atol=atol)

                mse_matrix[i, j] = result
                if i != j:
                    mse_matrix[j, i] = mse_matrix[i, j]

        return pd.DataFrame(mse_matrix, index=_simulators, columns=_simulators)

    def calculate_mse(self, a, b) -> float:
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


# -- SUPERVISOR: "Clearance" => All resources needed for db & file-storage access. Can act on behalf of the entity.
class Supervisor:
    def __init__(self, db_connector: MongoDbConnector, queue_timer: int = 20, preferred_queue_index: int = 0):
        self.db_connector = db_connector
        self.queue_timer = queue_timer
        self.preferred_queue_index = preferred_queue_index
        self.job_queue = self.db_connector.pending_jobs()
        self._supervisor_id: Optional[str] = "supervisor_" + unique_id()

    async def check_jobs(self, delay: int, n_attempts: int = 2) -> int:
        """Returns non-zero if max retries reached, zero otherwise."""

        # 1. For job (i) in job q, check if jobid exists for any job within db_connector.completed_jobs()
        # 1a. If so, pop the job from the pending queue
        # 2. If job doesnt yet exist in completed, summon a worker.
        # 3. Give the worker the pending job (i)
        # 4. Create completed job in which the job id from # 1 is the job id (id?) and results is worker.job_result
        # 5. Worker automatically is dismissed
        # 5a: TODO: In parallel, keep a pool of n workers List[Worker]. Summon them asynchronously and append more instances as demand increases.
        # 6. Sleep for a larger period of time
        # 7. At the end of check_jobs, run self.job_queue = self.db_connector.pending_jobs() (refresh)

        async def check():
            if len(self.job_queue):
                for i, pending_job in enumerate(self.job_queue):
                    # get job id
                    job_id = pending_job.get('job_id')
                    source = pending_job.get('path')

                    # check if job id exists in dbconn.completed
                    is_completed = self.job_exists(job_id=job_id, collection_name="completed_jobs")

                    if not is_completed:
                        # otherwise: create new worker with job
                        worker = Worker(job=pending_job)
                        result_data = await worker.run()

                        # when worker completes, dismiss worker (if in parallel) and create new completed job
                        completed_job_doc = await self.db_connector.insert_completed_job(
                            job_id=job_id,
                            results=result_data,
                            source=source
                        )

                    # job is complete, remove job from queue
                    self.job_queue.pop(i)

        for _ in range(n_attempts):
            await check()

            # sleep for a long period
            await sleep(10)

            # refresh job queue
            self.job_queue = self.db_connector.pending_jobs()

        return 0

    def job_exists(self, job_id: str, collection_name: str) -> bool:
        """Returns True if job with the given job_id exists, False otherwise."""
        unique_id_query = {'job_id': job_id}
        coll: MongoCollection = self.db_connector.db[collection_name]
        job = coll.find_one(unique_id_query) or None
        return job is not None

    # re-create loop here
    # def _handle_in_progress_job(self, job_exists: bool, job_id: str, comparison_id: str):
    #     if not job_exists:
    #         # print(f"In progress job does not yet exist for {job_comparison_id}")
    #         in_progress_job_id = unique_id()
    #         worker_id = unique_id()
    #         # id_kwargs = ['worker_id']
    #         # in_prog_kwargs = dict(zip(
    #         #     id_kwargs,
    #         #     list(map(lambda k: unique_id(), id_kwargs))
    #         # ))
    #         in_prog_kwargs = {'worker_id': worker_id, 'job_id': job_id, 'comparison_id': comparison_id}
    #         # in_prog_kwargs['comparison_id'] = job_comparison_id

    #         self.db_connector.insert_in_progress_job(**in_prog_kwargs)
    #         # print(f"Successfully created new progress job for {job_comparison_id}")
    #         # await supervisor.async_refresh_jobs()
    #     else:
    #         # print(f'In Progress Job for {job_comparison_id} already exists. Now checking if it has been completed.')
    #         pass

    #     return True

    # def _handle_completed_job(self, job_exists: bool, job_comparison_id: str, job_id: str, job_doc):
    #     if not job_exists:
    #         # print(f"Completed job does not yet exist for {job_comparison_id}")
    #         # pop in-progress job from internal queue and use it parameterize the worker
    #         in_prog_id = [job for job in self.db_connector.db.in_progress_jobs.find()].pop(self.preferred_queue_index)['job_id']

    #         # double-check and verify doc
    #         in_progress_doc = self.db_connector.db.in_progress_jobs.find_one({'job_id': in_prog_id})

    #         # generate new worker
    #         workers_id = in_progress_doc['worker_id']
    #         worker = self.call_worker(job_params=job_doc, worker_id=workers_id)

    #         # add the worker to the list of workers (for threadsafety)
    #         self.workers.insert(self.preferred_queue_index, worker.worker_id)

    #         # the worker returns the job result to the supervisor who saves it as part of a new completed job in the database
    #         completed_doc = self.db_connector.insert_completed_job(job_id=job_id, comparison_id=job_comparison_id, results=worker.job_result)

    #         # release the worker from being busy and refresh jobs
    #         self.workers.pop(self.preferred_queue_index)
    #         # await supervisor.async_refresh_jobs()
    #     else:
    #         pass

    #     return True

    # async def check_jobs(self, delay) -> int:
    #     """Returns non-zero if max retries reached, zero otherwise."""
    #     # if len(self.job_queue):
    #     #     for i, job in enumerate(self.job_queue):
    #     #         # get the next job in the queue based on the preferred_queue_index
    #     #         job_doc = self.job_queue.pop(self.preferred_queue_index)
    #     #         job_id = job_doc['job_id']
    #     #         job_comparison_id = job_doc['comparison_id']
    #     #         unique_id_query = {'job_id': job_id}
    #     #         in_progress_job = self.db_connector.db.in_progress_jobs.find_one(unique_id_query) or None
    #     #         _job_exists = partial(self._job_exists, job_id=job_id)
    #     #         # check for in progress job with same comparison id and make a new one if not
    #     #         # in_progress_exists = _job_exists(collection_name='in_progress_jobs', job_id=job_id)
    #     #         # self._handle_in_progress_job(in_progress_job, job_comparison_id)
    #     #         # self._handle_in_progress_job(job_exists=in_progress_exists, job_id=job_id, comparison_id=job_comparison_id)
    #     #         # do the same for completed jobs, which includes running the actual simulation comparison and returnin the results
    #     #         completed_exists = _job_exists(collection_name='completed_jobs', job_id=job_id)
    #     #         self._handle_completed_job(job_exists=completed_exists, job_comparison_id=job_comparison_id, job_doc=job_doc, job_id=job_id)
    #     #         # remove the job from queue
    #     #         # if len(job_queue):
    #     #         #     job_queue.pop(0)
    #     #     # sleep
    #     #     await sleep(delay)

    #     return 0
    


