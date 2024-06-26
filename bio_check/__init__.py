import uuid
import os
from asyncio import sleep

from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_DB_URI")


def unique_id(): return str(uuid.uuid4())


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
