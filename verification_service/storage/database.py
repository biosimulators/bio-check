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

from verification_service import unique_id
from verification_service.data_model.shared import BaseClass, MultipleConnectorError


@dataclass
class DatabaseConnector(ABC, BaseClass):
    """Abstract class that is both serializable and interacts with the database (of any type). """
    def __init__(self, connection_uri: str, database_id: str, connector_id: str):
        self.database_id = database_id
        self.client = self._get_client(connection_uri)
        self.db = self._get_database(self.database_id)

    @classmethod
    def timestamp(cls) -> str:
        return str(datetime.utcnow())

    @abstractmethod
    def _get_client(self, *args):
        pass

    @abstractmethod
    def _get_database(self, db_id: str):
        pass

    @abstractmethod
    def read(self, *args, **kwargs):
        pass

    @abstractmethod
    def write(self, *args, **kwargs):
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
    def __init__(self, connection_uri: str, database_id: str, connector_id: str = None):
        super().__init__(connection_uri, database_id, connector_id)
        self.pending_jobs = [j for j in self.db['pending_jobs'].find()]
        self.in_progress_jobs = self.db['in_progress_jobs']
        self.completed_jobs = self.db['completed_jobs']

    def _get_client(self, *args):
        return MongoClient(args[0])

    def _get_database(self, db_id: str) -> Database:
        return self.client.get_database(db_id)

    async def read(self, collection_name: str, **kwargs):
        """Args:
            collection_name: str
            kwargs: (as in mongodb query)
        """
        coll = self.get_collection(collection_name)
        result = await coll.find_one(kwargs)
        return result

    async def write(self, coll_name: str, **kwargs):
        """
            Args:
                coll_name: str: collection name in mongodb
                **kwargs: mongo db `insert_one` query defining the document where the key is as in the key of the document.
        """
        coll = self.get_collection(coll_name)
        result = coll.insert_one(kwargs)
        return result

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

    async def _job_exists(self, comparison_id: str, collection_name: str) -> bool:
        return self.__job_exists(comparison_id, collection_name)

    def __job_exists(self, comparison_id: str, collection_name: str) -> bool:
        coll = self.get_collection(collection_name)
        result = coll.find_one({'comparison_id': comparison_id}) is not None
        return result

    async def insert_pending_job(
            self,
            omex_path: str,
            simulators: List[str],
            comparison_id: str = None,
            ground_truth_report_path: str = None,
            include_outputs: bool = True,
    ) -> Union[Dict[str, str], Mapping[str, Any]]:
        pending_coll = self.get_collection("pending_jobs")


        specs_coll = self.get_collection("request_specs")
        results_coll = self.get_collection("results")


    async def _insert_pending_job(
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
        # coll = self.get_collection(collection_name)
        _time = self.timestamp()

        # check if query already exists
        # job_query = coll.find_one({"job_id": job_id})
        job_query = await self.read(collection_name, job_id=job_id)
        if isinstance(job_query, NoneType):
            pending_job_spec = {
                "job_id": job_id,
                "status": "PENDING",
                "omex_path": omex_path,
                "simulators": simulators,
                "comparison_id": comparison_id or f"uniform-time-course-comparison-{job_id}",
                "timestamp": _time,
                "ground_truth_report_path": ground_truth_report_path,
                "include_outputs": include_outputs
            }

            # coll.insert_one(pending_job_doc)
            pending_resp = await self.write(collection_name, **pending_job_spec)
            specs_resp = await self.write("request_specs", **pending_job_spec)

            return pending_job_spec
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
        # try each collection, starting with completed_jobs
        collections = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        for i, collection in enumerate(collections):
            coll = self.get_collection(collection)
            complete_job = coll.find_one({'comparison_id': comparison_id})
            if not isinstance(complete_job, NoneType):
                return complete_job
            else:
                next_i = i + 1 if i < len(collections) else i
                next_msg = collections[next_i] if next_i < len(collections) else "None"
                # TODO: Log this instead
                print(f"Job not found in {collection}. Now searching {collections[i + 1]}")
