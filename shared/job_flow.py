"""
Alex Patrie 1/6/2025

NOTE: This workflow is run by the microservices architecture and offloads ALL simulation logic to Biosimulator Processes!

The general workflow should be:

1. gateway: client uploads JSON spec file
2. gateway: gateway stores the JSON spec as a job in mongo
3. gateway: spec is returned to client as confirmation
4. worker: check for pending statuses in mongo collection (single)
5. worker: get simulators/deps from #4 TODO: somehow do this!!
6. worker: dynamic install #5
7. worker: change job status in DB to IN_PROGRESS
8. worker: run composition with pbg.Composite() and gather_results()
9. worker: change job status in DB to COMPLETE
10. worker: update job document ['results'] field with #8's data
11. worker: perhaps emit an event?
"""


import dotenv
import os
from typing import Any, Mapping, List

from process_bigraph import Composite
from bsp import app_registrar

from shared.data_model import DB_TYPE, DB_NAME, JOB_COLLECTION_NAME
from shared.database import MongoDbConnector


dotenv.load_dotenv("./.env")  # NOTE: create an env config at this filepath if dev

# establish common db connection
MONGO_URI = os.getenv("MONGO_URI")
db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)

# check jobs for most recent submission
all_jobs = db_connector.get_jobs()


class JobDispatcher(object):
    def __init__(self, connection_uri: str, database_id: str):
        self.db_connector = MongoDbConnector(connection_uri=connection_uri, database_id=database_id)

    @property
    def current_jobs(self) -> List[Mapping[str, Any]]:
        return self.db_connector.get_jobs()

    async def run(self):
        # iterate over all jobs
        for job in self.current_jobs:
            await self.process_job(job)

    async def process_job(self, job: Mapping[str, Any]):
        job_status = job["status"]
        if job_status.lower() != "pending":
            # 1. set job id
            job_id = job["job_id"]

            # 2. determine sims needed

            # 3. run dynamic env

            # 4. change job status to IN_PROGRESS

            # 5. from bsp import app_registrar.core

            # 6. create Composite() with core and job["job_spec"]

            # 7. run composition with instance from #6

            # 8. change status to COMPLETE

            # 9. update job in DB ['results'] to Composite().gather_results()



