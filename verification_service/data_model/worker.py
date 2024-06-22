# -- worker models -- #

from dataclasses import dataclass
from typing import *

from verification_service.data_model.shared import BaseClass, BaseModel, DbConnector, Job


@dataclass
class Worker(BaseClass):
    pending_job_ids: List[str]
    in_progress_job_ids: List[str]
    completed_job_ids: List[str]
    db_connector: DbConnector


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