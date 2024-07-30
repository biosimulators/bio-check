# -- This should serve as the main file for worker container -- #
import os
import tempfile
import uuid 
from asyncio import sleep
from dataclasses import dataclass
from functools import partial
from typing import *
from dotenv import load_dotenv

import numpy as np
import pandas as pd

from shared import BaseClass, MongoDbConnector, download_blob
from data_model import UtcComparison, SimulationError, UtcSpeciesComparison
from io_worker import get_sbml_species_names, get_sbml_model_file_from_archive, read_report_outputs
from output_data import generate_biosimulator_utc_outputs, _get_output_stack, sbml_output_stack, generate_sbml_utc_outputs, get_sbml_species_mapping

# for dev only
load_dotenv('../assets/.env_dev')

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

@dataclass
class Worker(BaseClass):
    job_params: Dict  # input arguments
    job_result: Optional[Dict] = None  # output result (utc_comparison.to_dict())
    worker_id: Optional[str] = unique_id()

    def __post_init__(self):
        input_fp = self.job_params['omex_path']
        if input_fp.endswith('.omex'):
            self._execute_omex_job()
        elif input_fp.endswith('.xml'):
            self._execute_sbml_job()

    def _execute_sbml_job(self):
        params = None
        out_dir = tempfile.mkdtemp()

        source_omex_blob_name = self.job_params['omex_path']
        local_fp = os.path.join(out_dir, source_omex_blob_name.split('/')[-1])
        download_blob(bucket_name=BUCKET_NAME, source_blob_name=source_omex_blob_name, destination_file_name=local_fp)

        try:
            simulators = self.job_params.get('simulators', [])
            include_outs = self.job_params.get('include_outputs', False)
            comparison_id = self.job_params['comparison_id']
            duration = self.job_params.get('duration', 10)
            n_steps = self.job_params.get('n_steps', 100)

            result = self.generate_sbml_utc_comparison(sbml_fp=local_fp, dur=duration, n_steps=n_steps)
            self.job_result = result
        except Exception as e:
            self.job_result = {"bio-check-message": f"Job for {self.job_params['comparison_id']} could not be completed because:\n{str(e)}"}

    def _execute_omex_job(self):
        params = None
        out_dir = tempfile.mkdtemp()

        # get omex from bucket
        source_omex_blob_name = self.job_params['omex_path']
        local_omex_fp = os.path.join(out_dir, source_omex_blob_name.split('/')[-1])
        download_blob(bucket_name=BUCKET_NAME, source_blob_name=source_omex_blob_name, destination_file_name=local_omex_fp)

        # get ground truth from bucket if applicable
        ground_truth_report_path = self.job_params['ground_truth_report_path']
        truth_vals = None
        if ground_truth_report_path is not None:
            source_report_blob_name = self.job_params['ground_truth_report_path']
            local_report_path = os.path.join(out_dir, ground_truth_report_path.split('/')[-1])
            download_blob(bucket_name=BUCKET_NAME, source_blob_name=source_report_blob_name, destination_file_name=local_report_path)
            truth_vals = read_report_outputs(local_report_path)

        try:
            simulators = self.job_params.get('simulators', [])
            include_outs = self.job_params.get('include_outputs', False)
            comparison_id = self.job_params['comparison_id']
            result = self.run_comparison_from_omex(
                omex_path=local_omex_fp,
                simulators=simulators,
                out_dir=out_dir,
                include_outputs=include_outs,
                comparison_id=comparison_id,
                truth_vals=truth_vals
            )
            self.job_result = result.model_dump()
        except Exception as e:
            self.job_result = {"bio-check-message": f"Job for {self.job_params['comparison_id']} could not be completed because:\n{str(e)}"}

    def run_comparison_from_omex(
            self,
            omex_path: str,
            simulators: List[str],
            out_dir: str,
            include_outputs: bool = True,
            comparison_id: str = None,
            truth_vals=None
    ) -> Union[UtcComparison, SimulationError]:
        """Execute a Uniform Time Course comparison for ODE-based simulators from Biosimulators."""
        # download the omex file from GCS
        # source_blob_name = omex_path.replace('gs://bio-check-requests-1', '')  # Assuming omex_fp is the blob name in GCS
        # local_omex_fp = os.path.join(out_dir, omex_path.split('/')[-1])
        # download_blob(bucket_name=BUCKET_NAME, source_blob_name=omex_path, destination_file_name=local_omex_fp)

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

        comparison = self.generate_omex_utc_comparison(
            omex_fp=omex_path,  # omex_path,
            out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
            simulators=simulators,
            comparison_id=comparison_id,
            ground_truth=ground_truth_data
        )

        # parse data for return vals
        spec_comparisons = []
        for spec_name, comparison_data in comparison['results'].items():
            species_comparison = UtcSpeciesComparison(
                mse=comparison_data['mse'],
                proximity=comparison_data['prox'],
                output_data=comparison_data.get('output_data') if include_outputs else {},
                species_name=spec_name)
            spec_comparisons.append(species_comparison)

        return UtcComparison(
            results=spec_comparisons,
            id=comparison_id,
            simulators=simulators)

    def generate_omex_utc_comparison(self, omex_fp, out_dir, simulators, comparison_id, ground_truth=None):
        model_file = get_sbml_model_file_from_archive(omex_fp, out_dir)
        sbml_species_names = get_sbml_species_names(model_file)
        results = {'results': {}, 'comparison_id': comparison_id}
        for i, species in enumerate(sbml_species_names):
            ground_truth_data = None
            if ground_truth:
                for data in ground_truth['data']:
                    if data['dataset_label'] == species:
                        ground_truth_data = data['data']
            results['results'][species] = self.generate_omex_utc_species_comparison(
                omex_fp=omex_fp,
                out_dir=out_dir,
                species_name=species,
                simulators=simulators,
                ground_truth=ground_truth_data
            )
        return results

    def generate_omex_utc_species_comparison(self, omex_fp, out_dir, species_name, simulators, ground_truth=None):
        output_data = generate_biosimulator_utc_outputs(omex_fp, out_dir, simulators)
        outputs = _get_output_stack(output_data, species_name)
        methods = ['mse', 'prox']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        for simulator_name in output_data.keys():
            for output in output_data[simulator_name]['data']:
                if output['dataset_label'] in species_name:
                    results['output_data'][simulator_name] = output['data'].tolist()
        return results

    def generate_sbml_utc_species_comparison(self, sbml_filepath, dur, n_steps, species_name, simulators=None, ground_truth=None):
        simulators = simulators or ['copasi', 'tellurium']
        if "amici" in simulators:
            simulators.remove("amici")

        output_data = generate_sbml_utc_outputs(sbml_fp=sbml_filepath, dur=dur, n_steps=n_steps)
        outputs = sbml_output_stack(species_name, output_data)
        methods = ['mse', 'prox']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        for simulator_name in output_data.keys():
            for spec_name, output in output_data[simulator_name].items():
                if species_name in spec_name:
                    results['output_data'][simulator_name] = output_data[simulator_name][spec_name].tolist()
        return results

    def generate_sbml_utc_comparison(self, sbml_fp, dur, n_steps):
        species_mapping = get_sbml_species_mapping(sbml_fp)
        results = {}
        for species_name in species_mapping.keys():
            species_comparison = self.generate_sbml_utc_species_comparison(sbml_fp, dur, n_steps, species_name)
            results[species_name] = species_comparison

        return results

    def _generate_species_comparison_matrix(
            self,
            outputs: Union[np.ndarray, List[np.ndarray]],
            simulators: List[str],
            method: Union[str, any] = 'prox',
            rtol: float = None,
            atol: float = None,
            ground_truth: np.ndarray = None
    ) -> pd.DataFrame:
        """Generate a Mean Squared Error comparison matrix of arr1 and arr2, indexed by simulators by default,
            or an AllClose Tolerance routine result if `method` is set to `prox`.

            Args:
                outputs: list of output arrays.
                simulators: list of simulator names.
                method: pass one of either: `mse` to perform a mean-squared error calculation
                    or `prox` to perform a pair-wise proximity tolerance test using `np.allclose(outputs[i], outputs[i+1])`.
                rtol:`float`: relative tolerance for comparison if `prox` is used.
                atol:`float`: absolute tolerance for comparison if `prox` is used.
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

        use_tol_method = method.lower() == 'prox'
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
@dataclass
class Supervisor(BaseClass):
    """
    # 1. Run check_jobs()
    # 2. Get an unassigned PENDING job.
    # 3. Mark #2 as IN_PROGRESS using the same comparison_id from #2
    # 4. Use #2 to give to worker as job_params
    # 4a. Associate #3 (in progress) with a worker
    # 5. Worker returns worker.job_result to the supervisor
    # 6. The supervisor (being the one with db access) then creates a new COMPLETED job doc with the output of #5.
    # 7. The supervisor stores the doc from #6.
    # 8. The return value of this is some sort of message(json?)
    """
    db_connector: MongoDbConnector  # TODO: Enable generic class
    queue_timer: Optional[float] = 5.0  # used for job check loop
    _supervisor_id: Optional[str] = unique_id()
    _instances = {}

    def __post_init__(self):
        # get dict of all jobs indexed by job ids
        self.workers = []
        self.job_queue = {}
        self.preferred_queue_index = 0
        self._refresh_jobs()

    def _refresh_jobs(self):
        self.jobs = self.get_jobs()
        self.pending_jobs = self.db_connector.pending_jobs()
        self.in_progress_jobs = self.db_connector.in_progress_jobs()
        self.completed_jobs = self.jobs['completed_jobs']

    async def refresh_jobs_async(self):
        return self._refresh_jobs()

    def get_jobs(self):
        coll_names = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        return dict(zip(
            coll_names,
            [[job for job in self.db_connector.db[coll_name].find()] for coll_name in coll_names])
        )

    # re-create loop here

    def _handle_in_progress_job(self, job_exists: bool, job_id: str, comparison_id: str):
        if not job_exists:
            # print(f"In progress job does not yet exist for {job_comparison_id}")
            in_progress_job_id = unique_id()
            worker_id = unique_id()
            # id_kwargs = ['worker_id']
            # in_prog_kwargs = dict(zip(
            #     id_kwargs,
            #     list(map(lambda k: unique_id(), id_kwargs))
            # ))
            in_prog_kwargs = {'worker_id': worker_id, 'job_id': job_id, 'comparison_id': comparison_id}
            # in_prog_kwargs['comparison_id'] = job_comparison_id

            self.db_connector.insert_in_progress_job(**in_prog_kwargs)
            # print(f"Successfully created new progress job for {job_comparison_id}")
            # await supervisor.async_refresh_jobs()
        else:
            # print(f'In Progress Job for {job_comparison_id} already exists. Now checking if it has been completed.')
            pass 

        return True

    def _handle_completed_job(self, job_exists: bool, job_comparison_id: str, job_id: str, job_doc):
        if not job_exists:
            # print(f"Completed job does not yet exist for {job_comparison_id}")
            # pop in-progress job from internal queue and use it parameterize the worker
            in_prog_id = [job for job in self.db_connector.db.in_progress_jobs.find()].pop(self.preferred_queue_index)['job_id']

            # double-check and verify doc
            in_progress_doc = self.db_connector.db.in_progress_jobs.find_one({'job_id': in_prog_id})

            # generate new worker
            workers_id = in_progress_doc['worker_id']
            worker = self.call_worker(job_params=job_doc, worker_id=workers_id)

            # add the worker to the list of workers (for threadsafety)
            self.workers.insert(self.preferred_queue_index, worker.worker_id)

            # the worker returns the job result to the supervisor who saves it as part of a new completed job in the database
            completed_doc = self.db_connector.insert_completed_job(job_id=job_id, comparison_id=job_comparison_id, results=worker.job_result)

            # release the worker from being busy and refresh jobs
            self.workers.pop(self.preferred_queue_index)
            # await supervisor.async_refresh_jobs()
        else:
            pass 

        return True

    def exists_in_next_queue(self, current_queue: list[dict], next_queue: list[dict]) -> bool:
        if len(current_queue) > 0:
            for current_job in current_queue:
                current_job_id = current_job['job_id']
                if len(next_queue) > 0:
                    for next_job in next_queue:
                        if current_job_id == next_job['job_id']:
                            return True

        return False

    async def check_jobs(self, max_retries=5, delay=5) -> int:
        """Returns non-zero if max retries reached, zero otherwise."""
        job_queue = self.db_connector.pending_jobs()
        n_tries = 0
        n_retries = 0

        if len(job_queue):
            # count tries
            n_tries += 1
            if n_tries == max_retries + 1:
                return 1

            # if n_tries is greater than 1 then it is a retry
            if n_tries > 1:
                n_retries += 1

            for i, job in enumerate(job_queue):
                # get the next job in the queue based on the preferred_queue_index
                job_doc = job_queue.pop(self.preferred_queue_index)
                job_id = job_doc['job_id']
                job_comparison_id = job_doc['comparison_id']
                unique_id_query = {'job_id': job_id}
                in_progress_job = self.db_connector.db.in_progress_jobs.find_one(unique_id_query) or None
                _job_exists = partial(self._job_exists, comparison_id=job_id)

                # check for in progress job with same comparison id and make a new one if not
                in_progress_exists = _job_exists(collection_name='in_progress_jobs')
                # self._handle_in_progress_job(in_progress_job, job_comparison_id)
                self._handle_in_progress_job(job_exists=in_progress_exists, job_id=job_id, comparison_id=job_comparison_id)
                # do the same for completed jobs, which includes running the actual simulation comparison and returnin the results
                completed_exists = _job_exists(collection_name='completed_jobs')
                self._handle_completed_job(job_exists=completed_exists, job_comparison_id=job_comparison_id, job_doc=job_doc, job_id=job_id)
                
                # remove the job from queue
                # if len(job_queue):
                #     job_queue.pop(0)

            # sleep
            await sleep(delay)
            job_queue = self.db_connector.pending_jobs()
        else:
            pass 
        return 0

    def _job_exists(self, **kwargs):
        unique_id_query = {'job_id': kwargs['job_id']}
        job = self.db_connector.db[kwargs['collection_name']].find_one(unique_id_query) or None
        return job is not None

    def call_worker(self, job_params: Dict, worker_id: Optional[str] = None) -> Worker:
        return Worker(job_params=job_params, worker_id=worker_id)
    


