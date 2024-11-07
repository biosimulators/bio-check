import os
import asyncio
import logging
from typing import List, Tuple

from process_bigraph import ProcessTypes
from dotenv import load_dotenv

from service.shared_worker import MongoDbConnector
from service.log_config import setup_logging
from service.job import Supervisor


def register_module(
        items_to_register: List[Tuple[str, str]],
        core: ProcessTypes,
        verbose=False
) -> None:
    for process_name, path in items_to_register:
        module_name, class_name = path.rsplit('.', 1)
        try:
            import_statement = f'worker.service.bigraph.{module_name}'

            module = __import__(
                 import_statement, fromlist=[class_name])

            # Get the class from the module
            bigraph_class = getattr(module, class_name)

            # Register the process
            core.process_registry.register(process_name, bigraph_class)
            print(f'Registered {process_name}') if verbose else None
        except Exception as e:
            print(f"Cannot register {class_name}. Error:\n**\n{e}\n**") if verbose else None
            continue


APP_PROCESS_REGISTRY = ProcessTypes()
IMPLEMENTATIONS = [
    ('output-generator', 'steps.OutputGenerator'),
    ('time-course-output-generator', 'steps.TimeCourseOutputGenerator'),
    # ('smoldyn_step', 'steps.SmoldynStep'),
    ('simularium_smoldyn_step', 'steps.SimulariumSmoldynStep'),
    ('mongo-emitter', 'steps.MongoDatabaseEmitter')
]
register_module(IMPLEMENTATIONS, APP_PROCESS_REGISTRY, verbose=True)


load_dotenv('../../assets/dev/config/.env_dev')

# logging TODO: implement this.
logger = logging.getLogger("biochecknet.worker.main.log")
setup_logging(logger)

# sleep params
DELAY_TIMER = 20
MAX_RETRIES = 30

# creds params
MONGO_URI = os.getenv("MONGO_URI")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
DB_NAME = "service_requests"

# shared db_connector
db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME)


def store_registered_addresses():
    # TODO: here, get the registered process addresses as a list and save it to mongo under process_registry
    registered_addresses = db_connector.get_registered_addresses()


async def main(max_retries=MAX_RETRIES):
    # set timeout counter
    n_retries = 0

    # create supervisor
    supervisor = Supervisor(db_connector=db_connector)
    # supervisor = CompositionSupervisor(db_connector=db_connector)

    while True:
        # no job has come in a while
        if n_retries == MAX_RETRIES:
            await asyncio.sleep(10)  # TODO: adjust this for client polling as needed

        await supervisor.check_jobs()
        await asyncio.sleep(5)
        n_retries += 1


if __name__ == "__main__":
    asyncio.run(main())
