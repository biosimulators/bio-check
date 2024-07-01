import os
import asyncio

from shared import MongoDbConnector
from jobs import Supervisor


DELAY_TIMER = 5
MAX_RETRIES = 5
MAX_TIMEOUTS = 2
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = "service_requests"


async def main():
    # keep in mind that data gets saved to ../../data

    # set timeout counter
    n_timeouts = 0

    # create connector and supervisor
    db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)
    supervisor = Supervisor(db_connector=db_connector)

    run = n_timeouts < MAX_TIMEOUTS
    while run:
        result = await supervisor.check_jobs(max_retries=MAX_RETRIES, delay=DELAY_TIMER)
        if result > 0:
            n_timeouts += 1


if __name__ == "__main__":
    asyncio.run(main())


# net=app-net
# docker network create "$net"
# docker run -d --rm --name "$lib" --net "$net" --platform linux/amd64 "$PKG_ROOT"-"$lib":latest
# docker run -it --name "$lib" --net "$net" --platform linux/amd64 "$PKG_ROOT"-"$lib"