import os
from tempfile import mkdtemp

from fastapi import UploadFile
from google.cloud import storage


def check_upload_file_extension(file: UploadFile, purpose: str, ext: str, message: str = None) -> bool:
    if not file.filename.endswith(ext):
        msg = message or f"Files for {purpose} must be passed in {ext} format."
        raise ValueError(msg)
    else:
        return True


async def save_uploaded_file(uploaded_file: UploadFile, save_dest: str) -> str:
    """Write `fastapi.UploadFile` instance passed by api gateway user to `save_dest`."""
    file_path = os.path.join(save_dest, uploaded_file.filename)
    with open(file_path, 'wb') as file:
        contents = await uploaded_file.read()
        file.write(contents)
    return file_path


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # The ID of your GCS bucket
    # bucket_name = "your-bucket-name"
    # The path to your file to upload
    # source_file_name = "local/path/to/file"
    # The ID of your GCS object
    # destination_blob_name = "storage-object-name"

    storage_client = storage.Client('bio-check-428516')
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    # Optional: set a generation-match precondition to avoid potential race conditions
    # and data corruptions. The request to upload is aborted if the object's
    # generation number does not match your precondition. For a destination
    # object that does not yet exist, set the if_generation_match precondition to 0.
    # If the destination object already exists in your bucket, set instead a
    # generation-match precondition using its generation number.
    generation_match_precondition = 0

    blob.upload_from_filename(source_file_name, if_generation_match=generation_match_precondition)

    return {
        'message': f"File {source_file_name} uploaded to {destination_blob_name}."
    }


async def write_uploaded_file(job_id: str, bucket_name: str, uploaded_file: UploadFile, extension: str) -> str:
    # bucket params
    upload_prefix = f"file_uploads/{job_id}/"
    bucket_prefix = f"gs://{bucket_name}/" + upload_prefix

    save_dest = mkdtemp()
    fp = await save_uploaded_file(uploaded_file, save_dest)  # save uploaded file to ephemeral store

    # Save uploaded omex file to Google Cloud Storage
    uploaded_file_location = None
    properly_formatted_omex = check_upload_file_extension(uploaded_file, 'uploaded_file', extension)
    if properly_formatted_omex:
        blob_dest = upload_prefix + fp.split("/")[-1]
        upload_blob(bucket_name=bucket_name, source_file_name=fp, destination_blob_name=blob_dest)
        uploaded_file_location = blob_dest

    return uploaded_file_location


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)

    # Download the file to a destination
    blob.download_to_filename(destination_file_name)


def download_file_from_bucket(source_blob_path: str, out_dir: str, bucket_name: str) -> str:
    """Download any file specified in a given job_params (mongo db collection document) which is saved in the bucket to out_dir. The file is assumed to originate from bucket_name.

        Returns:
            filepath (`str`) of the downloaded file.
    """
    source_blob_name = source_blob_path
    local_fp = os.path.join(out_dir, source_blob_name.split('/')[-1])
    download_blob(bucket_name=bucket_name, source_blob_name=source_blob_name, destination_file_name=local_fp)
    return local_fp
