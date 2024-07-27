import uuid
import os
from dotenv import load_dotenv
from tempfile import mkdtemp

from jobs import Worker
from shared import MongoDbConnector, save_uploaded_file, upload_blob

load_dotenv('../assets/.env_dev')

MONGO_URI = os.getenv('MONGO_URI')
GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
BUCKET_NAME = os.getenv('BUCKET_NAME')
DB_NAME = os.getenv('DB_NAME')


async def test_worker(uploaded_file: str, ground_truth_report: str = None):
    db_connector = MongoDbConnector(connection_uri=MONGO_URI, database_id=DB_NAME, connector_id="test_worker")
    job_id = str(uuid.uuid4())
    _time = db_connector.timestamp()

    # save_dest = mkdtemp()

    # bucket params
    upload_prefix = f"uploads/{job_id}/"
    bucket_prefix = f"gs://{BUCKET_NAME}/" + upload_prefix

    # Save uploaded omex file to Google Cloud Storage
    # omex_fp = await save_uploaded_file(uploaded_file, save_dest)

    # for local dev, the file is already written, duh!
    omex_fp = uploaded_file
    omex_blob_dest = upload_prefix + uploaded_file
    omex_path = bucket_prefix + uploaded_file
    upload_blob(BUCKET_NAME, omex_fp, omex_blob_dest)

    # Save uploaded reports file to Google Cloud Storage if applicable
    report_fp = None
    if ground_truth_report:
        report_fp = await save_uploaded_file(ground_truth_report, save_dest)
        report_blob_dest = upload_prefix + ground_truth_report.filename
        upload_blob(BUCKET_NAME, report_fp, report_blob_dest)
    report_path = bucket_prefix + ground_truth_report.filename if report_fp else None

    # run insert job
    pending_job_doc = await db_connector.insert_job_async(
        collection_name="pending_jobs",
        status="PENDING",
        job_id=job_id,
        omex_path=omex_path,
        simulators=simulators,
        comparison_id=comparison_id or f"uniform-time-course-comparison-{job_id}",
        timestamp=_time,
        ground_truth_report_path=report_path,
        include_outputs=include_outputs)

    # worker = Worker()
    # request specific params
