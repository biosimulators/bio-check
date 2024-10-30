import uuid
import os
import asyncio
from pprint import pp
from dotenv import load_dotenv
from tempfile import mkdtemp

# from shared import MongoDbConnector

load_dotenv('../assets/dev/.env_dev')

MONGO_URI = os.getenv('MONGO_URI')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BUCKET_NAME = os.getenv('BUCKET_NAME')
DB_NAME = os.getenv('DB_NAME')

# db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME, connector_id="test_worker")


def test_files_worker(test_queue_index=0):
    # get first job from queue
    # job = db_connector.pending_jobs().pop(test_queue_index)
    pass







