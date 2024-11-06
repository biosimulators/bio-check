import os
import pprint
from dotenv import load_dotenv

from process_bigraph import Composite

from worker.output_generator import CORE, generate_time_course_data
from worker.workers import VerificationWorker


load_dotenv('../assets/dev/config/.env_dev')

MONGO_URI = os.getenv('MONGO_URI')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BUCKET_NAME = os.getenv('BUCKET_NAME')
DB_NAME = os.getenv('DB_NAME')
TEST_SBML_FP = "/Users/alexanderpatrie/Desktop/repos/bio-check/assets/test_fixtures/sbml-core/BorisEJB.xml"


def test_files_worker(test_queue_index=0):
    # get first job from queue
    # job = db_connector.pending_jobs().pop(test_queue_index)
    pass


def test_generate_time_course_data():
    results = generate_time_course_data(
        input_fp=TEST_SBML_FP,
        start=0,
        end=1000,
        steps=5,
        simulators=['copasi', 'pysces', 'tellurium'],
        out_dir="../worker/outputs"
    )
    pprint.pp(results)

    return results


def test_sbml_comparison():
    worker = VerificationWorker({'job_id': 'verification-test'})
    result = worker._run_comparison_from_sbml(sbml_fp=TEST_SBML_FP, start=0, dur=1000, steps=5, simulators=['copasi', 'pysces', 'tellurium'])
    print(f'The worker result:\n{result}')
