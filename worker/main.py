import os
import asyncio

from shared import MongoDbConnector
from jobs import Supervisor


DELAY_TIMER = 5
MAX_RETRIES = 5
MAX_TIMEOUTS = 2
MONGO_URI = os.getenv("MONGO_URI")


# async def main():
#     keep in mind that data gets saved to ../../data
#     n_timeouts = 0
#     db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id="service_requests")
#     supervisor = Supervisor(db_connector=db_connector)
#     run = n_timeouts == MAX_TIMEOUTS
#     while run:
#         result = await supervisor.check_jobs(max_retries=MAX_RETRIES, delay=DELAY_TIMER)
#         if result > 0:
#             n_timeouts += 1


if __name__ == "__main__":
    for _ in range(5):
        print("Here is where we run")
        print(MONGO_URI)
        asyncio.run(asyncio.sleep(3))

