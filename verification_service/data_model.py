from typing import *
from datetime import datetime

from pydantic import BaseModel as _BaseModel, ConfigDict


# -- globally-used base model -- #

class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


class DatabaseStore(BaseModel):
    db_type: str  # i.e: mongo etc.
    client: Any


# -- api models -- #

class DbConnector(BaseModel):
    client: Any
    database_id: str


class DbClientResponse(BaseModel):
    message: str
    db_type: str  # ie: 'mongo', 'postgres', etc
    timestamp: str


class UtcComparisonRequestParams(BaseModel):
    simulators: List[str] = ["amici", "copasi", "tellurium"]
    include_output: Optional[bool] = True
    comparison_id: Optional[str] = None


class Job(BaseModel):
    id: str
    status: str
    results: Optional[Dict] = None


class FetchResultsResponse(BaseModel):
    content: Any


class PendingJob(BaseModel):
    id: str
    status: str = "PENDING"
    omex_path: str
    simulators: List[str]
    comparison_id: str
    timestamp: str


# -- worker models -- #

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

