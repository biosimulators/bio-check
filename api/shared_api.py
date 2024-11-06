# -- db connectors -- #
import logging
import os
from abc import abstractmethod, ABC
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
from typing import *

import dotenv
from google.cloud import storage
from pydantic import BaseModel as _BaseModel, ConfigDict
from fastapi import UploadFile
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database


# -- globally-shared content-- #

dotenv.load_dotenv("../assets/dev/config/.env_dev")

DB_TYPE = "mongo"  # ie: postgres, etc
DB_NAME = "service_requests"
BUCKET_NAME = os.getenv("BUCKET_NAME")


def file_upload_prefix(job_id: str):
    # bucket params
    upload_prefix = f"file_uploads/{job_id}/"
    bucket_prefix = f"gs://{BUCKET_NAME}/" + upload_prefix
    return upload_prefix, bucket_prefix


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


class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DatabaseCollections(Enum):
    PENDING_JOBS = "PENDING_JOBS".lower()
    IN_PROGRESS_JOBS = "IN_PROGRESS_JOBS".lower()
    COMPLETED_JOBS = "COMPLETED_JOBS".lower()


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


class DatabaseConnector(ABC):
    """Abstract class that is both serializable and interacts with the database (of any type). """
    def __init__(self, connection_uri: str, database_id: str, connector_id: str):
        self.database_id = database_id
        self.client = self._get_client(connection_uri)
        self.db = self._get_database(self.database_id)

    @staticmethod
    def timestamp() -> str:
        return str(datetime.utcnow())

    def refresh_collection(self, coll):
        for job in self.db[coll].find():
            self.db[coll].delete_one(job)

    def refresh_jobs(self):
        for collname in ['completed_jobs', 'in_progress_jobs', 'pending_jobs']:
            self.refresh_collection(collname)

    @abstractmethod
    def _get_client(self, *args):
        pass

    @abstractmethod
    def _get_database(self, db_id: str):
        pass

    @abstractmethod
    def pending_jobs(self):
        pass

    @abstractmethod
    def completed_jobs(self):
        pass

    @abstractmethod
    async def read(self, *args, **kwargs):
        pass

    @abstractmethod
    async def write(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_collection(self, **kwargs):
        pass


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

    def completed_jobs(self):
        return self._get_jobs_from_collection("completed_jobs")

    @property
    def data(self):
        return self._get_data()

    def _get_data(self):
        return {coll_name: [v for v in self.db[coll_name].find()] for coll_name in self.db.list_collection_names()}

    async def read(self, collection_name: DatabaseCollections | str, **kwargs):
        """Args:
            collection_name: str
            kwargs: (as in mongodb query)
        """
        coll_name = self._parse_enum_input(collection_name)
        coll = self.get_collection(coll_name)
        result = coll.find_one(kwargs.copy())
        return result

    async def write(self, collection_name: DatabaseCollections | str, **kwargs):
        """
            Args:
                collection_name: str: collection name in mongodb
                **kwargs: mongo db `insert_one` query defining the document where the key is as in the key of the document.
        """
        coll_name = collection_name

        coll = self.get_collection(coll_name)
        result = coll.insert_one(kwargs.copy())
        return kwargs

    def get_collection(self, collection_name: str) -> Collection:
        try:
            return self.db[collection_name]
        except:
            return None

    async def insert_job_async(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        return self.insert_job(collection_name, **kwargs)

    def insert_job(self, collection_name: str, **kwargs) -> Dict[str, Any]:
        coll = self.get_collection(collection_name)
        job_doc = kwargs.copy()
        print("Inserting job...")
        coll.insert_one(job_doc)
        print(f"Job successfully inserted: {self.db.pending_jobs.find_one(kwargs)}.")
        return kwargs

    async def update_job_status(self, collection_name: str, job_id: str, status: str | JobStatus):
        job_status = self._parse_enum_input(status)
        return self.db[collection_name].update_one({'job_id': job_id, }, {'$set': {'status': job_status}})

    def _parse_enum_input(self, _input: Any) -> str:
        return _input.value if isinstance(_input, Enum) else _input


