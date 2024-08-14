# -- db connectors -- #
import logging
import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import *

from google.cloud import storage
from pydantic import BaseModel as _BaseModel, ConfigDict
from fastapi import UploadFile
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


# -- globally-shared content-- #


DB_TYPE = "mongo"  # ie: postgres, etc
DB_NAME = "service_requests"
BUCKET_NAME = os.getenv("BUCKET_NAME")


def check_upload_file_extension(file: UploadFile, purpose: str, ext: str) -> bool:
    if not file.filename.endswith(ext):
        raise ValueError(f"Files for {purpose} must be passed in {ext} format.")
    else:
        return True


def setup_logging(fname: str):
    logging.basicConfig(
        filename=fname,
        level=logging.CRITICAL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client('bio-check-428516')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    return {
        'message': f"File {source_file_name} uploaded to {destination_blob_name}."
    }


def read_uploaded_file(bucket_name, source_blob_name, destination_file_name):
    download_blob(bucket_name, source_blob_name, destination_file_name)

    with open(destination_file_name, 'r') as f:
        return f.read()


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file to a destination
    blob.download_to_filename(destination_file_name)


def make_dir(fp: str):
    if not os.path.exists(fp):
        os.mkdir(fp)


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

    def _get_client(self, *args):
        return MongoClient(args[0])

    def _get_database(self, db_id: str) -> Database:
        return self.client.get_database(db_id)

    def _get_jobs_from_collection(self, coll_name: str):
        return [job for job in self.db[coll_name].find()]

    def pending_jobs(self):
        return self._get_jobs_from_collection("pending_jobs")

    def in_progress_jobs(self):
        return self._get_jobs_from_collection("in_progress_jobs")

    def completed_jobs(self):
        return self._get_jobs_from_collection("completed_jobs")

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

    def get_collection(self, collection_name: str) -> Collection:
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
            path: str,
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
            path: str,
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
        if isinstance(job_query, type(None)):
            pending_job_spec = {
                "job_id": job_id,
                "status": "PENDING",
                "path": path,
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

    def fetch_job(self, job_id: str) -> Mapping[str, Any]:
        # try each collection, starting with completed_jobs
        collections = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        for i, collection in enumerate(collections):
            coll = self.get_collection(collection)
            job = coll.find_one({'job_id': job_id})
            # case: job exists of some type for that comparison id; return that
            if not isinstance(job, type(None)):
                return job

        # case: no job exists for that id
        return {'bio-check-message': f"No job exists for the comparison id: {job_id}"}

    def refresh_jobs(self):
        def refresh_collection(coll):
            for job in self.db[coll].find():
                self.db[coll].delete_one(job)

        for collname in ['completed_jobs', 'in_progress_jobs', 'pending_jobs']:
            refresh_collection(collname)
