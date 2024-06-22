# -- worker models -- #

from dataclasses import dataclass
from typing import *

from verification_service.data_model.shared import BaseClass, BaseModel, DbConnector, Job


@dataclass
class Worker(BaseClass):
    db_connector: DbConnector
    pending_job_ids: List[str] = None
    in_progress_job_ids: List[str] = None
    completed_job_ids: List[str] = None

    def __post_init__(self):
        id_key = 'job_id'
        coll_names = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        job_ids = dict(zip(
            coll_names,
            [[job[id_key] for job in self.db_connector[coll_name].find()] for coll_name in coll_names]
        ))
        self.completed_job_ids = [job[id_key] for job in self.db_connector.db['completed_jobs'].find()]


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