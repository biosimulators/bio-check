import logging
from functools import partial
from asyncio import sleep
from typing import *
from dotenv import load_dotenv
from nbclient.client import timestamp
from pymongo.collection import Collection as MongoCollection

from shared import handle_exception
from workers import SimulationRunWorker, VerificationWorker, FilesWorker, CompositionWorker
from shared import BaseClass, MongoDbConnector, unique_id, JobStatus, DatabaseCollections
from log_config import setup_logging


# for dev only
load_dotenv('../assets/dev/.env_dev')


# logging
LOGFILE = "biochecknet_composer_worker_supervisor.log"
# logger = logging.getLogger(__name__)
# setup_logging(logger)


class Supervisor:
    def __init__(self, db_connector: MongoDbConnector, queue_timer: int = 10, preferred_queue_index: int = 0):
        self.db_connector = db_connector
        self.queue_timer = queue_timer
        self.preferred_queue_index = preferred_queue_index
        self.job_queue = self.db_connector.pending_jobs()
        self._supervisor_id: Optional[str] = "supervisor_" + unique_id()

    async def check_jobs(self) -> int:
        """Returns non-zero if max retries reached, zero otherwise."""

        # 1. For job (i) in job q, check if jobid exists for any job within db_connector.completed_jobs()
        # 1a. If so, pop the job from the pending queue
        # 2. If job doesnt yet exist in completed, summon a worker.
        # 3. Give the worker the pending job (i)
        # 4. Create completed job in which the job id from # 1 is the job id (id?) and results is worker.job_result
        # 5. Worker automatically is dismissed
        # 5a: TODO: In parallel, keep a pool of n workers List[Worker]. Summon them asynchronously and append more instances as demand increases.
        # 6. Sleep for a larger period of time
        # 7. At the end of check_jobs, run self.job_queue = self.db_connector.pending_jobs() (refresh)

        # async def check():
        #     for i, pending_job in enumerate(self.job_queue):
        #         # get job id
        #         job_id = pending_job.get('job_id')
        #         source = pending_job.get('path')
        #         source_name = source.split('/')[-1]
        #         # check if job id exists in dbconn.completed
        #         job_completed = self.job_exists(job_id=job_id, collection_name="completed_jobs")
        #         job_failed = self.job_exists(job_id=job_id, collection_name="failed_jobs")
        #         worker = None
        #         # case: job is not complete, otherwise do nothing
        #         if not job_completed and not job_failed:
        #             # check: run simulations
        #             if job_id.startswith('simulation-execution'):
        #                 worker = SimulationRunWorker(job=pending_job)
        #             # check: verifications
        #             elif job_id.startswith('verification'):
        #                 # otherwise: create new worker with job
        #                 worker = VerificationWorker(job=pending_job)
        #             # check: files
        #             elif job_id.startswith('files'):
        #                 worker = FilesWorker(job=pending_job)
        #             # check: composition
        #             elif job_id.startswith('composition-run'):
        #                 worker = CompositionWorker(job=pending_job)
        #             # change job status for client by inserting a new in progress job
        #             job_in_progress = self.job_exists(job_id=job_id, collection_name="in_progress_jobs")
        #             if not job_in_progress:
        #                 in_progress_job = await self.db_connector.insert_job_async(
        #                     collection_name=DatabaseCollections.IN_PROGRESS_JOBS.value,
        #                     job_id=job_id,
        #                     timestamp=self.db_connector.timestamp(),
        #                     status=JobStatus.IN_PROGRESS.value,
        #                     source=source_name
        #                 )
        #                 try:
        #                     # when worker completes, dismiss worker (if in parallel)
        #                     await worker.run()
        #                     # create new completed job using the worker's job_result TODO: refactor output nesting
        #                     result_data = worker.job_result
        #                     await self.db_connector.insert_job_async(
        #                         collection_name=DatabaseCollections.COMPLETED_JOBS.value,
        #                         job_id=job_id,
        #                         timestamp=self.db_connector.timestamp(),
        #                         status=JobStatus.COMPLETED.value,
        #                         results=result_data,
        #                         source=source_name
        #                     )
        #                 except:
        #                     # save new error to db
        #                     error = handle_exception('Job Error')
        #                     await self.db_connector.insert_job_async(
        #                         collection_name="failed_jobs",
        #                         job_id=job_id,
        #                         timestamp=self.db_connector.timestamp(),
        #                         status=JobStatus.FAILED.value,
        #                         results=error,
        #                         source=source_name
        #                     )

        for _ in range(self.queue_timer):
            # perform check
            await self._check()

            # rest
            await sleep(2)

            # refresh jobs
            self.job_queue = self.db_connector.pending_jobs()

        return 0

    async def _check(self):
        worker = None
        for i, pending_job in enumerate(self.job_queue):
            # get job params
            job_id = pending_job.get('job_id')
            source = pending_job.get('path')
            source_name = source.split('/')[-1]

            # check terminal collections for job
            job_completed = self.job_exists(job_id=job_id, collection_name="completed_jobs")
            job_failed = self.job_exists(job_id=job_id, collection_name="failed_jobs")

            # case: job is not complete, otherwise do nothing
            if not job_completed and not job_failed:
                # change job status for client by inserting a new in progress job
                job_in_progress = self.job_exists(job_id=job_id, collection_name="in_progress_jobs")
                if not job_in_progress:
                    in_progress_entry = {'job_id': job_id, 'timestamp': self.db_connector.timestamp(), 'status': JobStatus.IN_PROGRESS.value, 'source': source}
                    if job_id.startswith('composition-run'):
                        in_progress_entry['composite_spec'] = pending_job['composite_spec']
                        in_progress_entry['simulator'] = pending_job['simulator']
                        in_progress_entry['duration'] = pending_job['duration']

                    in_progress_job = await self.db_connector.insert_job_async(
                        collection_name="in_progress_jobs",
                        **in_progress_entry
                    )

                    # remove job from pending
                    self.db_connector.db.pending_jobs.delete_one({'job_id': job_id})

                # run job again
                try:
                    # check: run simulations

                    if job_id.startswith('simulation-execution'):
                        worker = SimulationRunWorker(job=pending_job)
                    # check: verifications
                    elif job_id.startswith('verification'):
                        worker = VerificationWorker(job=pending_job)
                    # check: files
                    elif job_id.startswith('files'):
                        worker = FilesWorker(job=pending_job)
                    # TODO: uncomment below to implement sse composition execution
                    # check: composition
                    # elif job_id.startswith('composition-run'):
                    #     worker = CompositionWorker(job=pending_job)
                    #     await worker.run(conn=self.db_connector)
                    #     result_data = worker.job_result
                    #     simulator = pending_job.get('simulator', 'copasi')
                    #     await self.db_connector.insert_job_async(
                    #         collection_name=DatabaseCollections.COMPLETED_JOBS.value,
                    #         job_id=job_id,
                    #         timestamp=self.db_connector.timestamp(),
                    #         status=JobStatus.COMPLETED.value,
                    #         source=source_name,
                    #         simulator=simulator,
                    #         results=result_data['data']
                    #     )

                    # when worker completes, dismiss worker (if in parallel)
                    await worker.run()
                    # create new completed job using the worker's job_result
                    result_data = worker.job_result
                    await self.db_connector.insert_job_async(
                        collection_name=DatabaseCollections.COMPLETED_JOBS.value,
                        job_id=job_id,
                        timestamp=self.db_connector.timestamp(),
                        status=JobStatus.COMPLETED.value,
                        results=result_data,
                        source=source_name,
                        requested_simulators=pending_job['simulators']
                    )
                    # remove in progress job
                    self.db_connector.db.in_progress_jobs.delete_one({'job_id': job_id})
                except:
                    # save new execution error to db
                    error = handle_exception('Job Execution Error')
                    await self.db_connector.insert_job_async(
                        collection_name="failed_jobs",
                        job_id=job_id,
                        timestamp=self.db_connector.timestamp(),
                        status=JobStatus.FAILED.value,
                        results=error,
                        source=source_name
                    )
                    # remove in progress job TODO: refactor this
                    self.db_connector.db.in_progress_jobs.delete_one({'job_id': job_id})

    def job_exists(self, job_id: str, collection_name: str) -> bool:
        """Returns True if job with the given job_id exists, False otherwise."""
        unique_id_query = {'job_id': job_id}
        coll: MongoCollection = self.db_connector.db[collection_name]
        job = coll.find_one(unique_id_query) or None

        return job is not None


class CompositionSupervisor:
    def __init__(self, db_connector: MongoDbConnector):
        self.db_connector = db_connector
        self.queue_timer = 5
        self.preferred_queue_index = 0
        self.job_queue = self.db_connector.pending_jobs()
        self._supervisor_id: Optional[str] = "supervisor_" + unique_id()

    def job_exists(self, job_id: str, collection_name: str) -> bool:
        """Returns True if job with the given job_id exists, False otherwise."""
        unique_id_query = {'job_id': job_id}
        coll: MongoCollection = self.db_connector.db[collection_name]
        job = coll.find_one(unique_id_query) or None

        return job is not None

    async def check_jobs(self) -> int:
        for _ in range(self.queue_timer):
            # perform check
            await self._check()

            # rest
            await sleep(2)

            # refresh jobs
            self.job_queue = self.db_connector.pending_jobs()

        return 0

    async def _check(self):
        worker = None
        for i, pending_job in enumerate(self.job_queue):
            # get job params
            job_id = pending_job.get('job_id')
            source = pending_job.get('path')
            source_name = source.split('/')[-1]
            duration = pending_job.get('duration')
            simulator = pending_job.get('simulator')

            # check terminal collections for job
            job_completed = self.job_exists(job_id=job_id, collection_name="completed_jobs")
            job_failed = self.job_exists(job_id=job_id, collection_name="failed_jobs")

            # case: job is not complete, otherwise do nothing
            if not job_completed and not job_failed:
                # run job again
                try:
                    # check: composition
                    if job_id.startswith('composition-run'):
                        worker = CompositionWorker(job=pending_job)

                    worker.run_composite_sse(duration)
                    result_data = worker.job_result

                    await self.db_connector.insert_job_async(
                        collection_name=DatabaseCollections.COMPLETED_JOBS.value,
                        job_id=job_id,
                        timestamp=self.db_connector.timestamp(),
                        status=JobStatus.COMPLETED.value,
                        source=source_name,
                        simulator=simulator,
                        results=result_data['data']
                    )
                except:
                    print('Error!')


