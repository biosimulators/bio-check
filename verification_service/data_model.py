from functools import partial
from typing import *
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel as _BaseModel, ConfigDict
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


# -- globally-used base model -- #

class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class BaseClass:
    def todict(self):
        return asdict(self)


@dataclass
class DbConnector(ABC, BaseClass):
    client: Any
    database_id: str
    db: Any = None

    def __post_init__(self):
        self.db = self._get_database(self.database_id)

    @classmethod
    def timestamp(cls) -> str:
        return str(datetime.utcnow())

    @abstractmethod
    def _get_database(self, db_id: str):
        pass

    @abstractmethod
    def get_collection(self, **kwargs):
        pass

    @abstractmethod
    def insert_job(self, collection_name: str):
        pass

    @abstractmethod
    def insert_pending_job(self, **kwargs):
        return self.insert_job(**kwargs)

    @abstractmethod
    def insert_in_progress_job(self, **kwargs):
        return self.insert_job(**kwargs)

    @abstractmethod
    def insert_completed_job(self, **kwargs):
        return self.insert_job(**kwargs)

    @abstractmethod
    def fetch_job(self, **kwargs):
        pass


@dataclass
class MongoDbConnector(DbConnector):
    client: MongoClient
    database_id: str
    db: Database = None

    def _get_database(self, db_id: str) -> Database:
        return self.client.get_database(db_id)

    def get_collection(self, collection_name: str) -> Union[Collection, None]:
        try:
            return self.db[collection_name]
        except:
            return None

    def insert_job(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        coll = self.get_collection(collection_name)
        job_doc = kwargs
        coll.insert_one(job_doc)
        return job_doc

    def insert_pending_job(
            self,
            job_id: str,
            omex_path: str,
            simulators: List[str],
            timestamp: str,
            comparison_id: str = None,
            reports_path: str = None
            ) -> Dict[str, str]:
        collection_name = "pending_jobs"
        coll = self.get_collection(collection_name)
        _time = self.timestamp()
        pending_job_doc = {
            "job_id": job_id,
            "status": "PENDING",
            "omex_path": omex_path,
            "simulators": simulators,
            "comparison_id": comparison_id or f"uniform-time-course-comparison-{job_id}",
            "timestamp": _time,
            "reports_path": reports_path or "null"}

        coll.insert_one(pending_job_doc)
        return pending_job_doc  # self.insert_job(collection_name=collection_name, **pending_job_doc)

    def insert_in_progress_job(self, job_id: str, comparison_id: str) -> Dict[str, str]:
        collection_name = "pending_jobs"
        _time = self.timestamp()
        in_progress_job_doc = {
            "job_id": job_id,
            "status": "IN_PROGRESS",
            "timestamp": _time,
            "comparison_id": comparison_id}

        return self.insert_job(collection_name=collection_name, **in_progress_job_doc)

    def insert_completed_job(self, job_id: str, comparison_id: str, results: Dict) -> Dict[str, str]:
        collection_name = "pending_jobs"
        _time = self.timestamp()
        in_progress_job_doc = {
            "job_id": job_id,
            "status": "COMPLETED",
            "timestamp": _time,
            "comparison_id": comparison_id,
            "results": results}

        return self.insert_job(collection_name=collection_name, **in_progress_job_doc)

    def fetch_job(self, client: MongoClient, job_id: str):
        """Check on the status and/or result of a given comparison run. This allows the user to poll status."""
        get_collection = partial(self.get_collection, client)
        coll_name = "completed_jobs"

        # look for completed job first
        coll: Collection = self.get_collection(coll_name)

        if not coll:
            in_progress_coll = self.get_collection("in_progress_jobs")
            if not in_progress_coll:
                # job is pending
                coll = self.get_collection("pending_jobs")
            else:
                # job is in progress
                coll = in_progress_coll

        return coll.find_one({"job_id": job_id})


# -- api models -- #


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


class PendingJob(BaseModel):
    job_id: str
    status: str = "PENDING"
    omex_path: str
    simulators: List[str]
    comparison_id: str
    timestamp: str
    reports_path: Optional[str] = None


class FetchResultsResponse(BaseModel):
    content: Any


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

