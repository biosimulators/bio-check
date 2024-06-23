# -- worker models -- #

from dataclasses import dataclass
from typing import *

from verification_service.data_model.shared import BaseClass, BaseModel, DbConnector, Job
from verification_service.worker.main import utc_comparison


@dataclass
class Worker(BaseModel):
    job_params: Dict  # input arguments
    job_result: Dict = None  # output result (utc_comparison.to_dict())

    async def __post_init__(self):
        result = await utc_comparison(**self.job_params)
        self.job_result = result.model_dump()


@dataclass
class Supervisor(BaseClass):
    db_connector: DbConnector
    jobs: Dict = None

    def __post_init__(self):
        id_key = 'job_id'
        coll_names = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        self.jobs = dict(zip(
            coll_names,
            [[job[id_key] for job in self.db_connector.db[coll_name].find()] for coll_name in coll_names]))

    def call_worker(self, job_params: Dict):
        # 1. Run check_jobs()
        # 2. Get an unassigned PENDING job.
        # 3. Mark #2 as IN_PROGRESS using the same comparison_id from #2
        # 4. Use #2 to give to worker as job_params
        # 5. Worker returns worker.job_result to the supervisor
        # 6. The supervisor (being the one with db access) then creates a new COMPLETED job doc with the output of #5.
        # 7. The supervisor stores the doc from #6.
        # 8. The return value of this is some sort of message(json?)
        return Worker(job_params=job_params)


class InProgressJob(Job):
    id: str
    status: str = "IN_PROGRESS"


class CompleteJob(Job):
    id: str
    results: Dict
    status: str = "COMPLETE"


class CustomError(BaseModel):
    detail: str


class ArchiveUploadResponse(BaseModel):
    filename: str
    content: str
    path: str


class UtcSpeciesComparison(BaseModel):
    species_name: str
    mse: Dict
    proximity: Dict
    output_data: Optional[Dict] = None


class UtcComparison(BaseModel):
    results: List[UtcSpeciesComparison]
    id: str
    simulators: List[str]


class SimulationError(Exception):
    def __init__(self, message: str):
        self.message = message


# api container fastapi, mongo database, worker container
class StochasticMethodError(BaseModel):
    message: str = "Only deterministic methods are supported."