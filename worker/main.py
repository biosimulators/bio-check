import os
import asyncio

from worker.shared import MongoDbConnector
from jobs import Supervisor


# sleep params
DELAY_TIMER = 5
MAX_RETRIES = 5
MAX_TIMEOUTS = 2

# creds params
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DB_NAME = "service_requests"

# shared db_connector
db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)


async def main():
    # set timeout counter
    n_timeouts = 0

    # create supervisor
    supervisor = Supervisor(db_connector=db_connector)

    # run async loop
    # run = n_timeouts < MAX_TIMEOUTS
    while True:
        result = await supervisor.check_jobs(max_retries=MAX_RETRIES, delay=DELAY_TIMER)
        if result > 0:
            n_timeouts += 1


if __name__ == "__main__":
    asyncio.run(main())


# net=app-net
# docker network create "$net"
# docker run -d --rm --name "$lib" --net "$net" --platform linux/amd64 "$PKG_ROOT"-"$lib":latest
# docker run -it --name "$lib" --net "$net" --platform linux/amd64 "$PKG_ROOT"-"$lib"