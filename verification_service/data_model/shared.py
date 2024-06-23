# -- globally-used base model -- #


from typing import *
from dataclasses import dataclass, asdict

from pydantic import BaseModel as _BaseModel, ConfigDict


class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class BaseClass:
    def todict(self):
        return asdict(self)


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
    include_output: Optional[bool] = True


class FetchResultsResponse(BaseModel):
    content: Any


from functools import partial
from types import NoneType
from typing import *
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime

from pydantic import BaseModel as _BaseModel, ConfigDict
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database





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
            reports_path: str = None,
            include_outputs: bool = True,
            ) -> Dict[str, str]:
        collection_name = "pending_jobs"
        coll = self.get_collection(collection_name)
        """
        omex_path: str,
                         simulators: list[str],
                         include_outputs: bool = True,
                         comparison_id: str | None = None,
                         ground_truth_report_path: str | None = None
                         
                        
        """
        _time = self.timestamp()
        pending_job_doc = {
            "job_id": job_id,
            "status": "PENDING",
            "omex_path": omex_path,
            "simulators": simulators,
            "comparison_id": comparison_id or f"uniform-time-course-comparison-{job_id}",
            "timestamp": _time,
            "ground_truth_report_path": reports_path,
            "include_outputs": include_outputs}

        coll.insert_one(pending_job_doc)
        return pending_job_doc  # self.insert_job(collection_name=collection_name, **pending_job_doc)

    def insert_in_progress_job(self, job_id: str, comparison_id: str, worker_id: str) -> Dict[str, str]:
        collection_name = "in_progress_jobs"
        _time = self.timestamp()
        in_progress_job_doc = {
            "job_id": job_id,
            "status": "IN_PROGRESS",
            "timestamp": _time,
            "comparison_id": comparison_id,
            "worker_id": worker_id}

        return self.insert_job(collection_name=collection_name, **in_progress_job_doc)

    def insert_completed_job(self, job_id: str, comparison_id: str, results: Any) -> Dict[str, str]:
        collection_name = "completed_jobs"
        _time = self.timestamp()
        in_progress_job_doc = {
            "job_id": job_id,
            "status": "COMPLETED",
            "timestamp": _time,
            "comparison_id": comparison_id,
            "results": results}

        return self.insert_job(collection_name=collection_name, **in_progress_job_doc)

    def fetch_job(self, comparison_id: str) -> Mapping[str, Any]:
        coll = self.db['jobs']
        status_decs = sorted(['COMPLETED', 'IN_PROGRESS', 'PENDING'])
        for status in status_decs:
            complete_job = coll.find_one({'status': status, 'comparison_id': comparison_id})
            if not isinstance(complete_job, NoneType):
                return complete_job
            else:
                print(f"Job not in {status}")