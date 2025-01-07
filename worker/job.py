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
import subprocess

import dotenv
import os
from typing import Any, Mapping, List

from process_bigraph import Composite

from shared.data_model import DB_TYPE, DB_NAME, JOB_COLLECTION_NAME
from shared.database import MongoDbConnector
from shared.dynamic_env import install_request_dependencies
from shared.log_config import setup_logging


logger = setup_logging(__file__)


class JobDispatcher(object):
    def __init__(self, connection_uri: str, database_id: str, timeout: int = 5):
        """
        :param connection_uri: mongodb connection URI
        :param database_id: mongodb database ID
        :param timeout: number of minutes for timeout. Default is 5 minutes
        """
        self.db_connector = MongoDbConnector(connection_uri=connection_uri, database_id=database_id)
        self.timeout = timeout * 60

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
            simulators = job["simulators"]

            # 3. try to run dynamic install
            try:
                installation_resp = install_request_dependencies(simulators)
            except subprocess.CalledProcessError as e:
                msg = f"Attempted installation for Job {job_id} was not successful."
                logger.error(msg)
                return {"job_id": job_id, "status": "FAILED", "result": msg}

            # 4. change job status to IN_PROGRESS
            # await self.db_connector.update_job_status(job_id=job_id, status="IN_PROGRESS")
            await self.db_connector.update_job(job_id=job_id, status="IN_PROGRESS")

            # 5. from bsp import app_registrar.core
            bsp = __import__("bsp")
            core = bsp.app_registrar.core

            # 6. create Composite() with core and job["job_spec"]
            composition = Composite(
                config={"state": job["spec"]},
                core=core
            )

            # 7. run composition with instance from #6 for specified duration (default 1)
            dur = job.get("duration", 1)
            composition.run(dur)

            # 8. get composition results indexed from ram-emitter
            results = composition.gather_results()[("emitter",)]

            # 9. update job in DB ['results'] to Composite().gather_results() AND change status to COMPLETE
            await self.db_connector.update_job(job_id=job_id, status="COMPLETE", results=results)





