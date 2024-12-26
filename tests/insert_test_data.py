import os
import asyncio

from shared_worker import MongoDbConnector


def __run_startup():
    db_connector = MongoDbConnector(connection_uri=os.environ['MONGO_URI'], database_id='service_requests')

    test_entry1 = {
        'job_id': 'test-entry1',
        'simulators': ['amici', 'copasi', 'pysces', 'tellurium']
    }

    test_entry2 = {
        'job_id': 'test-entry2',
        'simulators': ['smoldyn']
    }

    asyncio.run(db_connector.write(
        collection_name='pending_jobs',
        **test_entry1
    ))

    asyncio.run(db_connector.write(
        collection_name='pending_jobs',
        **test_entry2
    ))


__run_startup()
