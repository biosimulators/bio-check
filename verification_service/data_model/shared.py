# -- globally-shared models -- #


from typing import *
from dataclasses import dataclass, asdict

from pydantic import BaseModel as _BaseModel, ConfigDict


# -- base models --

class BaseModel(_BaseModel):
    """Base Pydantic Model with custom app configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class BaseClass:
    """Base Python Dataclass multipurpose class with custom app configuration."""
    def todict(self):
        return asdict(self)


# -- jobs --

class Job(BaseModel):
    job_id: str
    status: str
    timestamp: str
    comparison_id: str


class PendingJob(Job):
    job_id: str
    status: str
    timestamp: str
    comparison_id: str
    omex_path: str
    simulators: List[str]
    timestamp: str
    ground_truth_report_path: Optional[str] = None
    include_output: Optional[bool] = True


class InProgressJob(Job):
    job_id: str
    status: str
    timestamp: str
    comparison_id: str
    worker_id: str


class CompletedJob(Job):
    job_id: str
    status: str
    timestamp: str
    comparison_id: str
    results: Dict


class FetchResultsResponse(BaseModel):
    content: Any


