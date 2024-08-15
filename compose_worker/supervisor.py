import logging
from asyncio import sleep
from typing import *
from dotenv import load_dotenv
from pymongo.collection import Collection as MongoCollection

from workers import SimulationRunWorker, VerificationWorker
from shared import BaseClass, MongoDbConnector, unique_id
from log_config import setup_logging


# for dev only
load_dotenv('../assets/.env_dev')


# logging
LOGFILE = "biochecknet_composer_worker_supervisor.log"
# logger = logging.getLogger(__name__)
# setup_logging(logger)


class Supervisor:
    def __init__(self, db_connector: MongoDbConnector, queue_timer: int = 20, preferred_queue_index: int = 0):
        self.db_connector = db_connector
        self.queue_timer = queue_timer
        self.preferred_queue_index = preferred_queue_index
        self.job_queue = self.db_connector.pending_jobs()
        self._supervisor_id: Optional[str] = "supervisor_" + unique_id()

    async def check_jobs(self, delay: int, n_attempts: int = 2) -> int:
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

        async def check():
            if len(self.job_queue):
                for i, pending_job in enumerate(self.job_queue):
                    # get job id
                    job_id = pending_job.get('job_id')
                    source = pending_job.get('path')

                    # check if job id exists in dbconn.completed
                    is_completed = self.job_exists(job_id=job_id, collection_name="completed_jobs")

                    worker = None
                    if not is_completed:
                        # check: run simulations
                        if job_id.startswith('execute-simulations'):
                            worker = SimulationRunWorker(job=pending_job)
                        # check: verifications
                        elif job_id.startswith('verification'):
                            # otherwise: create new worker with job
                            worker = VerificationWorker(job=pending_job)

                        # when worker completes, dismiss worker (if in parallel) and create new completed job
                        result_data = await worker.run()
                        completed_job_doc = await self.db_connector.insert_completed_job(job_id=job_id, results=result_data, source=source)

                    # job is complete, remove job from queue
                    self.job_queue.pop(i)

        for _ in range(n_attempts):
            await check()

            # sleep for a long period
            await sleep(10)

            # refresh job queue
            self.job_queue = self.db_connector.pending_jobs()

        return 0

    def job_exists(self, job_id: str, collection_name: str) -> bool:
        """Returns True if job with the given job_id exists, False otherwise."""
        unique_id_query = {'job_id': job_id}
        coll: MongoCollection = self.db_connector.db[collection_name]
        job = coll.find_one(unique_id_query) or None
        return job is not None

    # re-create loop here
    # def _handle_in_progress_job(self, job_exists: bool, job_id: str, comparison_id: str):
    #     if not job_exists:
    #         # print(f"In progress job does not yet exist for {job_comparison_id}")
    #         in_progress_job_id = unique_id()
    #         worker_id = unique_id()
    #         # id_kwargs = ['worker_id']
    #         # in_prog_kwargs = dict(zip(
    #         #     id_kwargs,
    #         #     list(map(lambda k: unique_id(), id_kwargs))
    #         # ))
    #         in_prog_kwargs = {'worker_id': worker_id, 'job_id': job_id, 'comparison_id': comparison_id}
    #         # in_prog_kwargs['comparison_id'] = job_comparison_id

    #         self.db_connector.insert_in_progress_job(**in_prog_kwargs)
    #         # print(f"Successfully created new progress job for {job_comparison_id}")
    #         # await supervisor.async_refresh_jobs()
    #     else:
    #         # print(f'In Progress Job for {job_comparison_id} already exists. Now checking if it has been completed.')
    #         pass

    #     return True

    # def _handle_completed_job(self, job_exists: bool, job_comparison_id: str, job_id: str, job_doc):
    #     if not job_exists:
    #         # print(f"Completed job does not yet exist for {job_comparison_id}")
    #         # pop in-progress job from internal queue and use it parameterize the worker
    #         in_prog_id = [job for job in self.db_connector.db.in_progress_jobs.find()].pop(self.preferred_queue_index)['job_id']

    #         # double-check and verify doc
    #         in_progress_doc = self.db_connector.db.in_progress_jobs.find_one({'job_id': in_prog_id})

    #         # generate new worker
    #         workers_id = in_progress_doc['worker_id']
    #         worker = self.call_worker(job_params=job_doc, worker_id=workers_id)

    #         # add the worker to the list of workers (for threadsafety)
    #         self.workers.insert(self.preferred_queue_index, worker.worker_id)

    #         # the worker returns the job result to the supervisor who saves it as part of a new completed job in the database
    #         completed_doc = self.db_connector.insert_completed_job(job_id=job_id, comparison_id=job_comparison_id, results=worker.job_result)

    #         # release the worker from being busy and refresh jobs
    #         self.workers.pop(self.preferred_queue_index)
    #         # await supervisor.async_refresh_jobs()
    #     else:
    #         pass

    #     return True

    # async def check_jobs(self, delay) -> int:
    #     """Returns non-zero if max retries reached, zero otherwise."""
    #     # if len(self.job_queue):
    #     #     for i, job in enumerate(self.job_queue):
    #     #         # get the next job in the queue based on the preferred_queue_index
    #     #         job_doc = self.job_queue.pop(self.preferred_queue_index)
    #     #         job_id = job_doc['job_id']
    #     #         job_comparison_id = job_doc['comparison_id']
    #     #         unique_id_query = {'job_id': job_id}
    #     #         in_progress_job = self.db_connector.db.in_progress_jobs.find_one(unique_id_query) or None
    #     #         _job_exists = partial(self._job_exists, job_id=job_id)
    #     #         # check for in progress job with same comparison id and make a new one if not
    #     #         # in_progress_exists = _job_exists(collection_name='in_progress_jobs', job_id=job_id)
    #     #         # self._handle_in_progress_job(in_progress_job, job_comparison_id)
    #     #         # self._handle_in_progress_job(job_exists=in_progress_exists, job_id=job_id, comparison_id=job_comparison_id)
    #     #         # do the same for completed jobs, which includes running the actual simulation comparison and returnin the results
    #     #         completed_exists = _job_exists(collection_name='completed_jobs', job_id=job_id)
    #     #         self._handle_completed_job(job_exists=completed_exists, job_comparison_id=job_comparison_id, job_doc=job_doc, job_id=job_id)
    #     #         # remove the job from queue
    #     #         # if len(job_queue):
    #     #         #     job_queue.pop(0)
    #     #     # sleep
    #     #     await sleep(delay)

    #     return 0


