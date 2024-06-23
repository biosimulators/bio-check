# -- db connectors -- #


from abc import abstractmethod, ABC
from dataclasses import dataclass
from datetime import datetime
from types import NoneType
from typing import Mapping, Any, Dict, Union, List, Collection

from bson import ObjectId
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from verification_service.data_model.shared import BaseClass


@dataclass
class DatabaseConnector(ABC, BaseClass):
    """Abstract class that is both serializable and interacts with the database (of any type). """
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
class MongoDbConnector(DatabaseConnector):
    client: MongoClient
    database_id: str

    def __post_init__(self):
        self.db = self._get_database(self.database_id)
        self.pending_jobs = self.db['pending_jobs']
        self.in_progress_jobs = self.db['in_progress_jobs']
        self.completed_jobs = self.db['completed_jobs']

    def _get_database(self, db_id: str) -> Database:
        return self.client.get_database(db_id)

    def get_collection(self, collection_name: str) -> Collection[Mapping[str, Any] | Any] | None:
        try:
            return self.db[collection_name]
        except:
            return None

    def get_job(self, collection_name: str, query: dict):
        coll = self.get_collection(collection_name)
        job = coll.find_one(query)
        return job

    async def insert_job_async(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        return self.insert_job(collection_name, **kwargs)

    def insert_job(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        coll = self.get_collection(collection_name)
        job_doc = kwargs
        coll.insert_one(job_doc)
        return job_doc

    async def insert_pending_job(
            self,
            job_id: str,
            omex_path: str,
            simulators: List[str],
            timestamp: str,
            comparison_id: str = None,
            ground_truth_report_path: str = None,
            include_outputs: bool = True,
            ) -> Union[Dict[str, str], Mapping[str, Any]]:
        # get params
        collection_name = "pending_jobs"
        coll = self.get_collection(collection_name)
        _time = self.timestamp()

        # check if query already exists
        job_query = coll.find_one({"job_id": job_id})
        if isinstance(job_query, NoneType):
            pending_job_doc = {
                "job_id": job_id,
                "status": "PENDING",
                "omex_path": omex_path,
                "simulators": simulators,
                "comparison_id": comparison_id or f"uniform-time-course-comparison-{job_id}",
                "timestamp": _time,
                "ground_truth_report_path": ground_truth_report_path,
                "include_outputs": include_outputs}
            coll.insert_one(pending_job_doc)
            return pending_job_doc
        else:
            return job_query

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