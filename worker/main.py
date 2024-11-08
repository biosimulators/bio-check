import os
import asyncio
import logging
from typing import List, Tuple

from dotenv import load_dotenv

from shared_worker import MongoDbConnector
from log_config import setup_logging
from job import Supervisor
from bigraph_steps import BIGRAPH_ADDRESS_REGISTRY


# set up dev env if possible
load_dotenv('../assets/dev/config/.env_dev')

# logging
logger = logging.getLogger("biochecknet.worker.main.log")
setup_logging(logger)

# sleep params
DELAY_TIMER = 20
MAX_RETRIES = 30

# creds params
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DB_NAME = "service_requests"

# shared db_connector
db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)


async def store_registered_addresses(supervisor: Supervisor):
    # store list of process addresses that is available to the client via mongodb:
    confirmation = await supervisor.db_connector.write(
        collection_name="bigraph_registry",
        registered_addresses=BIGRAPH_ADDRESS_REGISTRY,
        timestamp=supervisor.db_connector.timestamp(),
        version="latest",
        return_document=True
    )
    return confirmation


async def main(max_retries=MAX_RETRIES):
    n_retries = 0
    supervisor = Supervisor(db_connector=db_connector)
    address_registration = await store_registered_addresses(supervisor)
    if not address_registration:
        logger.error("Failed to register addresses.")

    while True:
        # no job has come in a while
        if n_retries == MAX_RETRIES:
            await asyncio.sleep(10)  # TODO: adjust this for client polling as needed

        await supervisor.check_jobs()
        await asyncio.sleep(5)
        n_retries += 1


# if __name__ == "__main__":
#     asyncio.run(main())
