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
    def to_dict(self):
        return asdict(self)


class MultipleConnectorError(Exception):
    def __init__(self, message: str):
        self.message = message


# -- jobs --

class Job(BaseModel):
    job_id: str
    status: str
    timestamp: str
    comparison_id: str


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

