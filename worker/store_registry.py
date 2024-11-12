# creds params
import logging
import os
import asyncio

from dotenv import load_dotenv

from job import Supervisor
from shared_worker import MongoDbConnector
from log_config import setup_logging


load_dotenv('../assets/dev/config/.env_dev')

MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DB_NAME = "service_requests"

logger = logging.getLogger("biochecknet.worker.main.log")
setup_logging(logger)

db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)
supervisor = Supervisor(db_connector=db_connector)


if __name__ == '__main__':
    address_registration = asyncio.run(supervisor.store_registered_addresses())
    if not address_registration:
        logger.error("Failed to register addresses.")
    else:
        logger.info("Registered addresses.")
