from abc import abstractmethod, ABC
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import *

from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.results import UpdateResult

from shared.data_model import JOB_COLLECTION_NAME, WriteResponse


class DatabaseConnector(ABC):
    """Abstract class that is both serializable and interacts with the database (of any type). """
    def __init__(self, connection_uri: str, database_id: str, connector_id: str):
        self.database_id = database_id
        self.client = self._get_client(connection_uri)
        self.db = self._get_database(self.database_id)

    @property
    @abstractmethod
    def all_data(self):
        return None

    @abstractmethod
    def _get_client(self, *args):
        pass

    @abstractmethod
    def _get_database(self, db_id: str):
        pass

    @abstractmethod
    async def read(self, collection_name: str, *args, **kwargs):
        pass

    @abstractmethod
    async def write(self, collection_name: str, *args, **kwargs):
        pass

    @abstractmethod
    def get_jobs(self):
        pass

    @abstractmethod
    async def update_job_status(self, job_id: str, status: str):
        pass

    @abstractmethod
    def refresh_jobs(self):
        pass

    async def get_job(self, job_id: str, **kwargs):
        job_result = await self.read(collection_name=JOB_COLLECTION_NAME, job_id=job_id, **kwargs)
        return job_result

    @staticmethod
    def timestamp() -> str:
        return str(datetime.utcnow())


class MongoDbConnector(DatabaseConnector):
    def __init__(self, connection_uri: str, database_id: str, connector_id: str = None):
        super().__init__(connection_uri, database_id, connector_id)

    def get_collection(self, collection_name: str) -> Collection:
        # try:
        #     return self.db[collection_name]
        # except:
        #     return None
        return self.db[collection_name]

    @property
    def all_data(self):
        return {coll_name: [v for v in self.db[coll_name].find()] for coll_name in self.db.list_collection_names()}

    def _get_client(self, *args):
        return MongoClient(args[0])

    def _get_database(self, db_id: str) -> Database:
        return self.client.get_database(db_id)

    async def read(self, collection_name: str, **kwargs):
        """Args:
            collection_name: str
            kwargs: (as in mongodb query)
        """
        # coll_name = self._parse_enum_input(collection_name)
        coll = self.get_collection(collection_name)
        result = coll.find_one(kwargs.copy())
        return result

    async def write(self, collection_name: str, **kwargs):
        """
            Args:
                collection_name: str: collection name in mongodb
                **kwargs: mongo db `insert_one` query defining the document where the key is as in the key of the document. For example,
                    something like: results=, etc
        """
        try:
            coll = self.get_collection(collection_name)
            result = coll.insert_one(kwargs.copy())
            return kwargs.copy()
        except:
            return {}

    def get_jobs(self):
        coll = self.get_collection(JOB_COLLECTION_NAME)
        return [item for item in coll.find()]

    async def update_job_status(self, job_id: str, status: str) -> UpdateResult:
        coll = self.get_collection(JOB_COLLECTION_NAME)
        return coll.update_one(
            filter={'job_id': job_id},
            update={
                '$set': {
                    'status': status,
                    'last_updated': self.timestamp()
                }
            }
        )

    async def update_job(self, job_id: str, **params) -> UpdateResult:
        coll = self.get_collection(JOB_COLLECTION_NAME)
        job_params = params.copy()
        job_params['last_updated'] = self.timestamp()
        return coll.update_one(
            filter={'job_id': job_id},
            update={'$set': job_params}
        )

    def refresh_jobs(self):
        coll = JOB_COLLECTION_NAME
        for job in self.db[coll].find():
            self.db[coll].delete_one(job)
