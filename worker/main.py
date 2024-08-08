import os
import asyncio
import logging

from shared import MongoDbConnector, setup_logging
from jobs import Supervisor


# sleep params
DELAY_TIMER = 20
MAX_RETRIES = 20

# creds params
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DB_NAME = "service_requests"

# shared db_connector
db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)

setup_logging("biochecknet_worker_main.log")
logger = logging.getLogger(__name__)


async def main(max_retries=MAX_RETRIES):
    # set timeout counter
    n_retries = 0

    # create supervisor
    supervisor = Supervisor(db_connector=db_connector)

    while True:
        # no job has come in a while
        if n_retries == MAX_RETRIES:
            await asyncio.sleep(10)  # TODO: adjust this for client polling as needed

        await supervisor.check_jobs(delay=DELAY_TIMER)
        n_retries += 1


if __name__ == "__main__":
    asyncio.run(main())


# net=app-net
# docker network create "$net"
# docker run -d --rm --name "$lib" --net "$net" --platform linux/amd64 "$PKG_ROOT"-"$lib":latest
# docker run -it --name "$lib" --net "$net" --platform linux/amd64 "$PKG_ROOT"-"$lib"