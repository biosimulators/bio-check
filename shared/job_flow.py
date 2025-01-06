"""
A.P:
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

from shared.data_model import DB_TYPE, DB_NAME
from shared.database import MongoDbConnector


dotenv.load_dotenv("./.env")  # NOTE: create an env config at this filepath if dev

# establish common db connection
MONGO_URI = os.getenv("MONGO_URI")
db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)


# check jobs for most recent submission


# set job id


# determine sims needed


# run dynamic install


# change job status to IN_PROGRESS


# run composition with Composite()


# change status to COMPLETE


# update job in DB ['results'] to Composite().gather_results()



