import uuid
import os
import asyncio
from pprint import pp
from dotenv import load_dotenv
from tempfile import mkdtemp

from process_bigraph import Composite

from output_generator import CORE


load_dotenv('../assets/dev/config/.env_dev')

MONGO_URI = os.getenv('MONGO_URI')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BUCKET_NAME = os.getenv('BUCKET_NAME')
DB_NAME = os.getenv('DB_NAME')
TEST_SBML_FP = "/Users/alexanderpatrie/Desktop/repos/bio-check/assets/test_fixtures/sbml-core/BorisEJB.xml"


# db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME, connector_id="test_worker")


def test_files_worker(test_queue_index=0):
    # get first job from queue
    # job = db_connector.pending_jobs().pop(test_queue_index)
    pass


def test_utc_generator():
    doc = {
        'copasi': {
            '_type': 'step',
            'address': 'local:utc_output_generator',
            'config': {
                'input_file': TEST_SBML_FP,
                'simulator': 'copasi',
                'start_time': 0,
                'end_time': 10,
                'num_steps': 100
            },
            'inputs': {},
            'outputs': {
                'time_course_data': ['time_course_data_store'],
            }
        },
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'time_course_data': 'tree[float]'
                }
            },
            'inputs': {
                'time_course_data': ['time_course_data_store']
            }
        }
    }

    sim = Composite(
        config={
            'state': doc,
        },
        core=CORE
    )

    sim.save(filename="test_utc_output_generator_before.json", outdir="./outputs")
    sim.update({}, 1)
    results = sim.gather_results()
    print(f'Results:\n{results}')
    sim.save(filename="test_utc_output_generator_after.json", outdir="./outputs")
    return results


if __name__ == '__main__':
    test_utc_generator()





