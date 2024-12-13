import os
import uuid
from asyncio import sleep
from dataclasses import dataclass, asdict
from enum import Enum

from dotenv import load_dotenv
# from biosimulators_processes import CORE


# -- globally-shared content-- #


load_dotenv('../assets/dev/config/.env_dev')

DB_TYPE = "mongo"  # ie: postgres, etc
DB_NAME = "service_requests"
BUCKET_NAME = os.getenv("BUCKET_NAME")


# -- shared functions -- #




def unique_id():
    return str(uuid.uuid4())


def handle_exception(context: str) -> str:
    import traceback
    from pprint import pformat
    tb_str = traceback.format_exc()
    error_message = pformat(f"{context} error:\n{tb_str}")
    
    return error_message


async def load_arrows(timer):
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


# -- base python dataclass with to_dict() method -- #

@dataclass
class BaseClass:
    """Base Python Dataclass multipurpose class with custom app configuration."""
    def to_dict(self):
        return asdict(self)


# -- jobs -- #

class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DatabaseCollections(Enum):
    PENDING_JOBS = "PENDING_JOBS".lower()
    IN_PROGRESS_JOBS = "IN_PROGRESS_JOBS".lower()
    COMPLETED_JOBS = "COMPLETED_JOBS".lower()


# -- database connectors: currently exclusive to mongodb. TODO: create a dbconnector for a relational db -- #

