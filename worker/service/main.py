import os
import asyncio
import logging

from dotenv import load_dotenv

from service import APP_PROCESS_REGISTRY
from service.shared_worker import MongoDbConnector
from service.log_config import setup_logging
from service.job import Supervisor


load_dotenv('../../assets/dev/config/.env_dev')

# logging TODO: implement this.
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


def store_registered_addresses():
    # TODO: here, get the registered process addresses as a list and save it to mongo under process_registry
    pass


async def main(max_retries=MAX_RETRIES):
    # set timeout counter
    n_retries = 0

    # create supervisor
    supervisor = Supervisor(db_connector=db_connector)
    # supervisor = CompositionSupervisor(db_connector=db_connector)

    while True:
        # no job has come in a while
        if n_retries == MAX_RETRIES:
            await asyncio.sleep(10)  # TODO: adjust this for client polling as needed

        await supervisor.check_jobs()
        await asyncio.sleep(5)
        n_retries += 1


if __name__ == "__main__":
    asyncio.run(main())
