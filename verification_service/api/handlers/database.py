from typing import *
from functools import partial
from datetime import datetime

from fastapi import FastAPI
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient
from pymongo.database import Database as MongoDatabase
from pymongo.results import InsertOneResult

from verification_service.data_model import DatabaseStore, PendingJob


# TODO: Add mongo interaction here.

def timestamp():
    return str(datetime.utcnow())


def get_mongo_database(client: MongoClient, db_name: str) -> MongoDatabase:
    return client.get_database(db_name)


def get_mongo_collection(db: MongoDatabase, collection_name: str) -> Union[Collection, None]:
    try:
        return db[collection_name]
    except:
        return None


def get_service_collection(client: MongoClient, collection_name: str) -> Collection:
    db = get_mongo_database(client, "service_requests")
    return get_mongo_collection(db, collection_name)


def insert_pending_job(
        client: MongoClient,
        job_id: str,
        omex_path: str,
        simulators: List[str],
        timestamp: str,
        comparison_id: str = None,
        reports_path: str = None
        ) -> Dict[str, str]:
    coll = get_service_collection(client, "pending_jobs")
    pending_job_doc = {
        "job_id": job_id,
        "status": "PENDING",
        "omex_path": omex_path,
        "simulators": simulators,
        "comparison_id": comparison_id or f"uniform-time-course-comparison-{job_id}",
        "timestamp": timestamp,
        "reports_path": reports_path or "null"
    }

    # create job record in MongoDB
    coll.insert_one(pending_job_doc)

    return pending_job_doc


def fetch_comparison_job(client: MongoClient, job_id: str):
    """Check on the status and/or result of a given comparison run. This allows the user to poll status."""
    get_collection = partial(get_service_collection, client)
    coll_name = "completed_jobs"

    # look for completed job first
    coll: Collection = get_collection(coll_name)

    if not coll:
        in_progress_coll = get_collection("in_progress_jobs")
        if not in_progress_coll:
            # job is pending
            coll = get_collection("pending_jobs")
        else:
            # job is in progress
            coll = in_progress_coll

    return coll.find_one({"job_id": job_id})
