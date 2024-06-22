import tempfile
import os
import uuid
from typing import *

from fastapi import HTTPException
from pymongo.mongo_client import MongoClient

from verification_service.worker.compare import (
    generate_biosimulators_utc_comparison)
from verification_service.worker.io import read_report_outputs
from verification_service.data_model import UtcSpeciesComparison, UtcComparison, SimulationError, MongoDbConnector

DB_TYPE = "mongo"  # ie: postgres, etc
DB_NAME = "service_requests"

MONGO_URI = os.getenv("MONGO_DB_URI")
mongo_client = MongoClient(MONGO_URI)
db_connector = MongoDbConnector(client=mongo_client, database_id=DB_NAME)


def jobid(): return str(uuid.uuid4())


async def utc_comparison(
        omex_path: str,
        simulators: List[str],
        include_outputs: bool = True,
        comparison_id: str = None,
        ground_truth_report_path: str = None
        ) -> Union[UtcComparison, SimulationError]:
    try:
        out_dir = tempfile.mktemp()
        truth_vals = read_report_outputs(ground_truth_report_path) if ground_truth_report_path is not None else None

        comparison_id = comparison_id or 'biosimulators-utc-comparison'
        comparison = await generate_biosimulators_utc_comparison(
            omex_fp=omex_path,
            out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
            simulators=simulators,
            comparison_id=comparison_id,
            ground_truth=truth_vals)

        spec_comparisons = []
        for spec_name, comparison_data in comparison['results'].items():
            species_comparison = UtcSpeciesComparison(
                mse=comparison_data['mse'],
                proximity=comparison_data['prox'],
                output_data=comparison_data.get('output_data') if include_outputs else {},
                species_name=spec_name)
            spec_comparisons.append(species_comparison)
    except Exception as e:
        return SimulationError(str(e))

    return UtcComparison(
        results=spec_comparisons,
        id=comparison_id,
        simulators=simulators)


async def check_jobs(db_connector: MongoDbConnector):
    jobs = [job for job in db_connector.db['pending_jobs'].find()]

    jobs_to_process = []
    while len(jobs) > 0:
        # populate the queue of jobs with params to be processed
        job = jobs.pop(0)
        jobs_to_process.append(job)

        # mark the job in progress before handing it off
        in_progress_job_id = jobid()
        in_progress_doc = db_connector.insert_in_progress_job(in_progress_job_id, job['comparison_id'])
        print(f"Successfully marked document IN_PROGRESS:\n{in_progress_doc}")

        job_result = await utc_comparison(
            omex_path=job['omex_path'],
            simulators=job['simulators'],
            comparison_id=job['comparison_id'],
        )

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
