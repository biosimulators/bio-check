import os

from bio_check.database import MongoDbConnector
from bio_check.worker.jobs import Supervisor

DELAY_TIMER = 5
MAX_RETRIES = 5
MAX_TIMEOUTS = 2
MONGO_URI = os.getenv("MONGO_URI")


async def main():
    n_timeouts = 0
    run = n_timeouts >= MAX_TIMEOUTS
    db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id="service_requests")
    supervisor = Supervisor(db_connector=db_connector)
    while run:
        result = await supervisor.check_jobs(max_retries=MAX_RETRIES, delay=DELAY_TIMER)
        if result > 0:
            n_timeouts += 1
