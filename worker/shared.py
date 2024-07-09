# -- db connectors -- #

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

    print(
        f"File {source_file_name} uploaded to {destination_blob_name}."
    )


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


async def save_uploaded_file(uploaded_file: UploadFile, save_dest: str) -> str:
    # TODO: replace this with s3 and use save_dest
    file_path = os.path.join(save_dest, uploaded_file.filename)
    with open(file_path, 'wb') as file:
        contents = await uploaded_file.read()
        file.write(contents)
    return file_path


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
        if isinstance(job_query, type(None)):
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
            if not isinstance(complete_job, type(None)):
                return complete_job
            else:
                next_i = i + 1 if i < len(collections) else i
                next_msg = collections[next_i] if next_i < len(collections) else "None"
                # TODO: Log this instead
                print(f"Job not found in {collection}. Now searching {collections[i + 1]}")
