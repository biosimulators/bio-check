import os
import uuid
from types import NoneType
import tempfile
from dataclasses import dataclass, field
from functools import partial
from time import sleep
from types import NoneType
from typing import *

from pymongo.mongo_client import MongoClient

from verification_service.worker.compare import utc_comparison
from verification_service.storage.database import MongoDbConnector
from verification_service.data_model.worker import UtcComparison, SimulationError


import numpy as np
import pandas as pd

from biosimulator_processes.execute import exec_utc_comparison

from verification_service import unique_id
from verification_service.data_model.shared import BaseClass
from verification_service.storage.database import MongoDbConnector
from verification_service.data_model.worker import UtcComparison, SimulationError, UtcSpeciesComparison, cascading_load_arrows
from verification_service.io import get_sbml_species_names, get_sbml_model_file_from_archive, read_report_outputs
from verification_service.worker.output_data import generate_biosimulator_utc_outputs, _get_output_stack


DB_TYPE = "mongo"  # ie: postgres, etc
DB_NAME = "service_requests"
MONGO_URI = os.getenv("MONGO_DB_URI")
mongo_client = MongoClient(MONGO_URI)
db_connector = MongoDbConnector(client=mongo_client, database_id=DB_NAME)


def jobid(): return str(uuid.uuid4())




@dataclass
class Worker(BaseClass):
    job_params: Dict  # input arguments
    job_result: Optional[Dict] = None  # output result (utc_comparison.to_dict())
    worker_id: Optional[str] = unique_id()

    def execute_job(self):
        """pop job_id, status, timestamp"""
        params = self.job_params.copy()
        list(map(lambda k: params.pop(k), ['job_id', 'status', 'timestamp', '_id']))
        result = self._run_comparison(**params)
        self.job_result = result.model_dump()

    def _run_comparison(
            self,
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
        comparison = self._generate_utc_comparison(
            omex_fp=omex_path,
            out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
            simulators=simulators,
            comparison_id=comparison_id,
            ground_truth=truth_vals.to_dict() if not isinstance(truth_vals, NoneType) else truth_vals)
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

    def _generate_utc_comparison(self, omex_fp, out_dir, simulators, comparison_id, ground_truth=None):
        model_file = get_sbml_model_file_from_archive(omex_fp, out_dir)
        sbml_species_names = get_sbml_species_names(model_file)
        results = {'results': {}, 'comparison_id': comparison_id}
        for i, species in enumerate(sbml_species_names):
            ground_truth_data = None
            if ground_truth:
                for data in ground_truth['data']:
                    if data['dataset_label'] == species:
                        ground_truth_data = data['data']
            results['results'][species] = self._generate_utc_species_comparison(
                omex_fp=omex_fp,
                out_dir=out_dir,
                species_name=species,
                simulators=simulators,
                ground_truth=ground_truth_data
            )
        return results

    def _generate_utc_species_comparison(self, omex_fp, out_dir, species_name, simulators, ground_truth=None):
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
                result = self.calculate_mse(output_i, output_j) if method_type == 'mse' else compare_arrays(arr1=output_i, arr2=output_j, rtol=rtol, atol=atol)

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
    jobs: Optional[Dict] = None  # comparison ids  TODO: change this?
    queue_timer: Optional[float] = 5.0  # used for job check loop

    def __post_init__(self):
        # get dict of all jobs indexed by job ids
        id_key = 'job_id'
        coll_names = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        self.workers = []
        self.job_queue = {}
        self.preferred_queue_index = 0
        self._refresh_jobs()

    def _refresh_jobs(self):
        self.jobs = self.get_jobs()

    def get_jobs(self, id_key: str = 'job_id'):
        coll_names = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        return dict(zip(
            coll_names,
            [[job for job in self.db_connector.db[coll_name].find()] for coll_name in coll_names])
        )

    def initialize(self):
        # activate job queue
        run = True
        while True:
            self.__setattr__('job_queue', self.check_jobs())
            if not len(self.job_queue['pending_jobs']):
                run = False
            else:
                cascading_load_arrows(self.queue_timer)  # sleep(self.queue_timer)

    def _job_exists(self, collection_name: str, comparison_id: str):
        unique_id_query = {'comparison_id': comparison_id}
        job = self.db_connector.db[collection_name].find_one(unique_id_query) or None
        return not isinstance(job, NoneType)

    def check_jobs(self) -> Dict[str, str]:
        jobs_to_complete = self.jobs['pending_jobs']

        # case: uncompleted/pending jobs exist
        if len(jobs_to_complete):
            in_progress_jobs = self.jobs['in_progress_jobs']
            preferred_queue_index = self.preferred_queue_index  # TODO: How can we make this more robust/dynamic?

            for job in jobs_to_complete:
                # get the next job in the queue based on the preferred_queue_index
                job_id = jobs_to_complete.pop(preferred_queue_index)
                job_doc = self.db_connector.db.pending_jobs.find_one({'job_id': job_id})
                job_comparison_id = job_doc['comparison_id']
                unique_id_query = {'comparison_id': job_comparison_id}
                in_progress_job = self.db_connector.db.in_progress_jobs.find_one(unique_id_query) or None

                job_exists = partial(self._job_exists, comparison_id=job_comparison_id)
                # case: the job (which has a unique comparison_id) has not been picked up and thus no in-progress job for the given comparison id yet exists
                if not job_exists('in_progress_jobs'):
                    in_progress_job_id = unique_id()
                    worker_id = unique_id()
                    id_kwargs = ['job_id', 'worker_id']
                    in_prog_kwargs = dict(zip(
                        id_kwargs,
                        list(map(lambda k: unique_id(), id_kwargs))
                    ))
                    self.db_connector.insert_in_progress_job(**in_prog_kwargs, comparison_id=job_comparison_id)
                    self._refresh_jobs()

                # check to see if for some reason the completed job is already there and call worker exec if not
                if not job_exists('completed_jobs'):
                    # pop in-progress job from internal queue and use it parameterize the worker
                    in_prog_id = in_progress_jobs.pop(preferred_queue_index)
                    in_progress_doc = self.db_connector.db.in_progress_jobs.find_one({'job_id': in_prog_id})
                    workers_id = in_progress_doc['worker_id']
                    worker = self.call_worker(job_params=job_doc, worker_id=workers_id)

                    # add the worker to the list of workers (for threadsafety)
                    self.workers.insert(preferred_queue_index, worker)

                    # the worker returns the job result to the supervisor who saves it as part of a new completed job in the database
                    completed_doc = self.db_connector.insert_completed_job(job_id=unique_id(), comparison_id=job_comparison_id, results=worker.job_result)

                    # release the worker from being busy and refresh jobs
                    self.workers.pop(preferred_queue_index)
                    self._refresh_jobs()

            return self.jobs.copy()

    def call_worker(self, job_params: Dict, worker_id: Optional[str] = None) -> Worker:
        return Worker(job_params=job_params, worker_id=worker_id)



async def exec_utc_comparison(
        omex_path: str,
        simulators: List[str],
        include_outputs: bool = True,
        comparison_id: str = None,
        ground_truth_report_path: str = None
        ) -> Union[UtcComparison, SimulationError]:
    """Execute a Uniform Time Course comparison for ODE-based simulators from Biosimulators."""
    result = await utc_comparison(
        omex_path=omex_path,
        simulators=simulators,
        include_outputs=include_outputs,
        comparison_id=comparison_id,
        ground_truth_report_path=ground_truth_report_path
    )
    return result


async def check_jobs():
    jobs = [job for job in db_connector.db['pending_jobs'].find()]

    jobs_to_process = []
    while len(jobs) > 0:
        # populate the queue of jobs with params to be processed
        job = jobs.pop(0)
        jobs_to_process.append(job)

        comparison_id = job['comparison_id']

        # check if in progress and mark the job in progress before handing it off if not

        in_progress_coll = db_connector.get_collection("in_progress_jobs")
        in_progress_job = in_progress_coll.find_one({'comparison_id': comparison_id})

        if isinstance(in_progress_job, NoneType):
            in_progress_job_id = jobid()
            in_progress_doc = db_connector.insert_in_progress_job(in_progress_job_id, job['comparison_id'])
            print(f"Successfully marked document IN_PROGRESS:\n{in_progress_doc}")

            # TODO: Make this work with storage.
            # job_result: UtcComparison = await utc_comparison(
            #     omex_path=job['omex_path'],
            #     simulators=job['simulators'],
            #     comparison_id=comparison_id,
            # )
        else:
            print(f"In progress job already exists: {in_progress_job}")

        completed_id = jobid()
        completed_doc = db_connector.insert_completed_job(
            job_id=completed_id,
            comparison_id=in_progress_job['comparison_id'],
            # results=job_result.model_dump())
            results={"a": [1, 2, 4]})  # here result would be that generated in job result

        # rest to check
        from asyncio import sleep
        print(f"Sleeping for {5}...zzzzzzz...")
        await sleep(5)
    return {"status": "all jobs completed."}

# @app.post(
#     "/utc-comparison",
#     response_model=UtcComparison,
#     name="Uniform Time Course Comparison.",
#     operation_id="utc-comparison",
#     summary="Compare UTC outputs for each species in a given model file. You may pass either a model file or OMEX archive file.")
# async def utc_comparison(
#         uploaded_file: UploadFile = File(...),
#         simulators: List[str] = Query(default=['amici', 'copasi', 'tellurium']),
#         include_outputs: bool = Query(default=True),
#         comparison_id: str = Query(default=None),
#         # ground_truth: List[List[float]] = None,
#         # time_course_config: Dict[str, Union[int, float]] = Body(default=None)
# ) -> UtcComparison:
#     out_dir = tempfile.mkdtemp()
#     save_dir = tempfile.mkdtemp()
#     omex_path = await save_uploaded_file(uploaded_file, save_dir)
#     comparison_name = comparison_id or f'api-generated-utc-comparison-for-{simulators}'
#     # generate async comparison
#     comparison = generate_utc_comparison(
#         omex_fp=omex_path,
#         simulators=simulators,
#         include_outputs=include_outputs,
#         comparison_id=comparison_name)
#     spec_comparisons = []
#     for spec_name, comparison_data in comparison['results'].items():
#         species_comparison = UtcSpeciesComparison(
#             mse=comparison_data['mse'],
#             proximity=comparison_data['prox'],
#             output_data=comparison_data.get('output_data'),
#             species_name=spec_name)
#         spec_comparisons.append(species_comparison)
#     return UtcComparison(results=spec_comparisons, id=comparison_name, simulators=simulators)


# @app.post(
#     "/utc-comparison",  # "/biosimulators-utc-comparison",
#     response_model=UtcComparison,
#     name="Biosimulator Uniform Time Course Comparison",
#     operation_id="biosimulators-utc-comparison",
#     summary="Compare UTC outputs from Biosimulators for a model from a given archive.")
# async def utc_comparison(
#         uploaded_file: UploadFile = File(..., description="OMEX/COMBINE Archive File."),
#         simulators: List[str] = Query(
#             default=['amici', 'copasi', 'tellurium'],
#             description="Simulators to include in the comparison."
#         ),
#         include_outputs: bool = Query(
#             default=True,
#             description="Whether to include the output data on which the comparison is based."
#         ),
#         comparison_id: str = Query(
#             default=None,
#             description="Descriptive identifier for this comparison."
#         ),
#         ground_truth_report: UploadFile = File(
#             default=None,
#             description="reports.h5 file defining the so-called ground-truth to be included in the comparison.")
#         ) -> UtcComparison:
#
#     try:
#         save_dir = tempfile.mkdtemp()
#         out_dir = tempfile.mkdtemp()
#         omex_path = await save_uploaded_file(uploaded_file, save_dir)
#
#         if ground_truth_report is not None:
#             report_filepath = await save_uploaded_file(ground_truth_report, save_dir)
#             ground_truth = await read_report_outputs(report_filepath)
#             truth_vals = ground_truth.to_dict()['data']
#             # d = [d.to_dict() for d in ground_truth.data if "time" not in d.dataset_label.lower()]
#             # truth_vals = [data['data'].tolist() for data in d]
#         else:
#             truth_vals = None
#
#         comparison_id = comparison_id or 'biosimulators-utc-comparison'
#         comparison = await generate_biosimulators_utc_comparison(
#             omex_fp=omex_path,
#             out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
#             simulators=simulators,
#             comparison_id=comparison_id,
#             ground_truth=truth_vals)
#
#         spec_comparisons = []
#         for spec_name, comparison_data in comparison['results'].items():
#             species_comparison = UtcSpeciesComparison(
#                 mse=comparison_data['mse'],
#                 proximity=comparison_data['prox'],
#                 output_data=comparison_data.get('output_data') if include_outputs else {},
#                 species_name=spec_name)
#             spec_comparisons.append(species_comparison)
#     except SimulationError as e:
#         raise HTTPException(status_code=400, detail=str(e.message))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
#
#     return UtcComparison(
#         results=spec_comparisons,
#         id=comparison_id,
#         simulators=simulators)


# @app.post(
#     "/biosimulators-utc-species-comparison",
#     response_model=UtcSpeciesComparison,
#     summary="Compare UTC outputs from Biosimulators for a given species name")
# async def biosimulators_utc_species_comparison(
#         uploaded_file: UploadFile = File(...),
#         species_id: str = Query(...),
#         simulators: List[str] = Query(default=['amici', 'copasi', 'tellurium']),
#         include_outputs: bool = Query(default=True)
# ) -> UtcSpeciesComparison:
#     # handle os structures
#     save_dir = tempfile.mkdtemp()
#     out_dir = tempfile.mkdtemp()
#     omex_path = os.path.join(save_dir, uploaded_file.filename)
#     with open(omex_path, 'wb') as file:
#         contents = await uploaded_file.read()
#         file.write(contents)
#     # generate async comparison
#     comparison = await generate_biosimulators_utc_species_comparison(
#         omex_fp=omex_path,
#         out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
#         species_name=species_id,
#         simulators=simulators)
#     out_data = comparison['output_data'] if include_outputs else None
#     return UtcSpeciesComparison(
#         mse=comparison['mse'],
#         proximity=comparison['prox'],
#         output_data=out_data,
#         species_name=species_id)
