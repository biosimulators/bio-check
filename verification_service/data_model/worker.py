# -- worker models -- #


import uuid
from asyncio import sleep
from dataclasses import dataclass
from types import NoneType
from typing import *

from verification_service.data_model.shared import BaseClass, BaseModel, DbConnector, MongoDbConnector, Job
from verification_service.worker.main import utc_comparison


def jobid(): return str(uuid.uuid4())


@dataclass
class Worker(BaseModel):
    job_params: Dict  # input arguments
    job_result: Dict = None  # output result (utc_comparison.to_dict())

    async def __post_init__(self):
        result = await utc_comparison(**self.job_params)
        self.job_result = result.model_dump()


@dataclass
class Supervisor(BaseClass):
    db_connector: MongoDbConnector  # TODO: Enable generic class
    jobs: Dict = None  # comparison ids  TODO: change this?
    job_queue: Dict[str, str] = None  # returns the status of the job check
    check_timer: float = 5.0

    async def __post_init__(self):
        # get dict of all jobs indexed by comparison ids
        id_key = 'comparison_id'
        coll_names = ['completed_jobs', 'in_progress_jobs', 'pending_jobs']
        self.jobs = dict(zip(
            coll_names,
            [[job[id_key] for job in self.db_connector.db[coll_name].find()] for coll_name in coll_names]))

        # activate job queue
        self.job_queue = await self._check_jobs()

    async def _check_jobs(self) -> Dict[str, str]:
        try:
            # jobs = [job for job in self.db_connector.db['pending_jobs'].find()]
            jobs_to_complete = self.jobs['pending_jobs']

            while len(jobs_to_complete) > 0:
                # populate the queue of jobs with params to be processed
                pending_comparison_id = jobs_to_complete.pop(0)
                pending_job = self.db_connector.db.pending_jobs.find_one({'comparison_id': pending_comparison_id})

                comparison_id = pending_job['comparison_id']

                # check if in progress and mark the job in progress before handing it off if not

                in_progress_coll = self.db_connector.get_collection("in_progress_jobs")
                in_progress_job = in_progress_coll.find_one({'comparison_id': comparison_id})

                if isinstance(in_progress_job, NoneType):
                    in_progress_job_id = jobid()
                    in_progress_doc = self.db_connector.insert_in_progress_job(in_progress_job_id, pending_job['comparison_id'])
                    print(f"Successfully marked comparison IN_PROGRESS:\n{in_progress_doc['comparison_id']}\n")
                else:
                    print(f"Comparison already in progress: {in_progress_job['comparison_id']}\n")

                # generate worker result and create a new completed job
                worker = await self.call_worker(pending_job)
                completed_id = jobid()
                completed_doc = self.db_connector.insert_completed_job(
                    job_id=completed_id,
                    comparison_id=in_progress_job['comparison_id'],
                    results=worker.job_result
                )

                # sleep with fancy logging :)
                print(f"Sleeping for {self.check_timer}")
                await cascading_load_arrows(self.check_timer)

            # job is finished and successfully complete
            status = "all jobs completed."
        except Exception as e:
            status = f"something went wrong:\n{e}"

        return {"status": status}

    async def call_worker(self, job_params: Dict):
        # 1. Run check_jobs()
        # 2. Get an unassigned PENDING job.
        # 3. Mark #2 as IN_PROGRESS using the same comparison_id from #2
        # 4. Use #2 to give to worker as job_params
        # 5. Worker returns worker.job_result to the supervisor
        # 6. The supervisor (being the one with db access) then creates a new COMPLETED job doc with the output of #5.
        # 7. The supervisor stores the doc from #6.
        # 8. The return value of this is some sort of message(json?)
        return await Worker(job_params=job_params)


class InProgressJob(Job):
    id: str
    status: str = "IN_PROGRESS"


class CompleteJob(Job):
    id: str
    results: Dict
    status: str = "COMPLETE"


class CustomError(BaseModel):
    detail: str


class ArchiveUploadResponse(BaseModel):
    filename: str
    content: str
    path: str


class UtcSpeciesComparison(BaseModel):
    species_name: str
    mse: Dict
    proximity: Dict
    output_data: Optional[Dict] = None


class UtcComparison(BaseModel):
    results: List[UtcSpeciesComparison]
    id: str
    simulators: List[str]


class SimulationError(Exception):
    def __init__(self, message: str):
        self.message = message


# api container fastapi, mongo database, worker container
class StochasticMethodError(BaseModel):
    message: str = "Only deterministic methods are supported."


async def cascading_load_arrows(timer):
    check_timer = timer
    ell = ""
    bars = ""
    msg = "|"
    n_ellipses = timer
    log_interval = check_timer / n_ellipses
    for n in range(n_ellipses):
        single_interval = log_interval / 3
        await sleep(single_interval)
        bars += "="
        disp = bars + ">"
        if n == n_ellipses - 1:
            disp += "|"
        print(disp)

