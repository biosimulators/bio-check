import tempfile
from dataclasses import dataclass
from types import NoneType
from typing import *

import numpy as np
import pandas as pd

from biosimulator_processes.execute import exec_utc_comparison

from verification_service import unique_id
from verification_service.data_model.shared import BaseClass, MongoDbConnector
from verification_service.data_model.worker import UtcComparison, SimulationError, UtcSpeciesComparison, cascading_load_arrows
from verification_service.io import get_sbml_species_names, get_sbml_model_file_from_archive, read_report_outputs
from verification_service.worker.output_data import generate_biosimulator_utc_outputs, _get_output_stack


@dataclass
class Worker(BaseClass):
    job_params: Dict  # input arguments
    job_result: Optional[Dict] = None  # output result (utc_comparison.to_dict())
    worker_id: Optional[str] = unique_id()

    def __post_init__(self):
        """pop job_id, status, timestamp"""
        params = self.job_params.copy()
        map(lambda k: params.pop(k), ['job_id', 'status', 'timestamp'])
        result = utc_comparison(**params)
        self.job_result = result.model_dump()


@dataclass
class Supervisor(BaseClass):
    db_connector: MongoDbConnector  # TODO: Enable generic class
    jobs: Optional[Dict] = None  # comparison ids  TODO: change this?
    job_queue: Optional[Dict[str, str]] = None  # returns the status of the job check
    check_timer: Optional[float] = 5.0

    def __post_init__(self):
        # get dict of all jobs indexed by comparison ids
        id_key = 'job_id'
        coll_names = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        self.jobs = dict(zip(
            coll_names,
            [[job[id_key] for job in self.db_connector.db[coll_name].find()] for coll_name in coll_names]))
        self.job_queue = {}

    def initialize(self):
        # activate job queue
        self.job_queue = self._check_jobs()

    def _check_jobs(self) -> Dict[str, str]:
        try:
            # jobs = [job for job in self.db_connector.db['pending_jobs'].find()]
            jobs_to_complete = self.jobs['pending_jobs'].copy()

            while len(jobs_to_complete) > 0:
                # populate the queue of jobs with params to be processed
                pending_job_id = jobs_to_complete.pop(0)
                pending_job = self.db_connector.db.pending_jobs.find_one({'job_id': pending_job_id})

                comparison_id = pending_job['comparison_id']

                # check if in progress and mark the job in progress before handing it off if not

                in_progress_coll = self.db_connector.get_collection("in_progress_jobs")
                in_progress_job = in_progress_coll.find_one({'comparison_id': comparison_id})

                # in progress job does not yet exist for the given pending job
                if isinstance(in_progress_job, NoneType):
                    # summon worker
                    worker = self.call_worker(pending_job)

                    # create and store an in-progress job
                    in_progress_job_id = unique_id()
                    in_progress_doc = self.db_connector.insert_in_progress_job(
                        job_id=in_progress_job_id,
                        comparison_id=comparison_id,
                        worker_id=worker.worker_id
                    )

                    completed_id = unique_id()
                    completed_doc = self.db_connector.insert_completed_job(
                        job_id=completed_id,
                        comparison_id=comparison_id,
                        results=worker.job_result
                    )

                    # sleep with fancy logging :)
                    print(f"Sleeping for {self.check_timer}")
                    cascading_load_arrows(self.check_timer)
                    print(f"Successfully marked comparison IN_PROGRESS:\n{in_progress_doc['comparison_id']}\n")
                else:
                    print(f"Comparison already in progress and is probably also complete: {in_progress_job['comparison_id']}\n")

            # job is finished and successfully complete
            status = "all jobs completed."
        except Exception as e:
            status = f"something went wrong:\n{e}"

        return {"status": status}

    def call_worker(self, job_params: Dict):
        # 1. Run check_jobs()
        # 2. Get an unassigned PENDING job.
        # 3. Mark #2 as IN_PROGRESS using the same comparison_id from #2
        # 4. Use #2 to give to worker as job_params
        # 4a. Associate #3 (in progress) with a worker
        # 5. Worker returns worker.job_result to the supervisor
        # 6. The supervisor (being the one with db access) then creates a new COMPLETED job doc with the output of #5.
        # 7. The supervisor stores the doc from #6.
        # 8. The return value of this is some sort of message(json?)
        return Worker(job_params=job_params)


def utc_comparison(
        omex_path: str,
        simulators: List[str],
        include_outputs: bool = True,
        comparison_id: str = None,
        ground_truth_report_path: str = None
        ) -> Union[UtcComparison, SimulationError]:
    """Execute a Uniform Time Course comparison for ODE-based simulators from Biosimulators."""
    out_dir = tempfile.mktemp()
    truth_vals = read_report_outputs(ground_truth_report_path) if ground_truth_report_path is not None else None
    comparison_id = comparison_id or 'biosimulators-utc-comparison'
    comparison = generate_biosimulators_utc_comparison(
        omex_fp=omex_path,
        out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
        simulators=simulators,
        comparison_id=comparison_id,
        ground_truth=truth_vals.to_dict())
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


def generate_utc_comparison(omex_fp: str, simulators: list[str], comparison_id: str = None, include_outputs: bool = True):
    # TODO: ensure that specific simulators get selected with the list of simulators.
    return exec_utc_comparison(
        omex_fp=omex_fp,
        simulators=simulators,
        comparison_id=comparison_id or 'utc-simulator-verification',
        include_outputs=include_outputs)


def generate_biosimulators_utc_species_comparison(omex_fp, out_dir, species_name, simulators, ground_truth=None):
    output_data = generate_biosimulator_utc_outputs(omex_fp, out_dir, simulators)
    outputs = _get_output_stack(output_data, species_name)
    methods = ['mse', 'prox']
    results = dict(zip(
        methods,
        list(map(
            lambda m: generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth).to_dict(),
            methods
        ))
    ))
    results['output_data'] = {}
    for simulator_name in output_data.keys():
        for output in output_data[simulator_name]['data']:
            if output['dataset_label'] in species_name:
                results['output_data'][simulator_name] = output['data'].tolist()
    return results


def generate_biosimulators_utc_comparison(omex_fp, out_dir, simulators, comparison_id, ground_truth=None):
    model_file = get_sbml_model_file_from_archive(omex_fp, out_dir)
    sbml_species_names = get_sbml_species_names(model_file)
    results = {'results': {}, 'comparison_id': comparison_id}
    for i, species in enumerate(sbml_species_names):
        ground_truth_data = None
        if ground_truth:
            for data in ground_truth['data']:
                if data['dataset_label'] == species:
                    ground_truth_data = data['data']
        results['results'][species] = generate_biosimulators_utc_species_comparison(
            omex_fp=omex_fp,
            out_dir=out_dir,
            species_name=species,
            simulators=simulators,
           #  ground_truth=ground_truth[i] if isinstance(ground_truth, list or np.ndarray) else None)
            ground_truth=ground_truth_data)
    return results


def calculate_mse(a, b) -> float:
    if isinstance(a, list):
        a = np.array(a)
    if isinstance(b, list):
        b = np.array(b)
    return np.mean((a - b) ** 2)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, atol=None, rtol=None) -> bool:
    """Original methodology copied from biosimulations runutils."""
    max1 = max(arr1)
    max2 = max(arr2)
    aTol = atol or max(1e-3, max1*1e-5, max2*1e-5)
    rTol = rtol or 1e-4
    return np.allclose(arr1, arr2, rtol=rTol, atol=aTol)


def generate_species_comparison_matrix(
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
            result = calculate_mse(output_i, output_j) if method_type == 'mse' else compare_arrays(arr1=output_i, arr2=output_j, rtol=rtol, atol=atol)

            mse_matrix[i, j] = result
            if i != j:
                mse_matrix[j, i] = mse_matrix[i, j]

    return pd.DataFrame(mse_matrix, index=_simulators, columns=_simulators)
