import os
import re
import unicodedata
import urllib
import zipfile
from pathlib import Path
from tempfile import mkdtemp
from typing import *

import libsbml
import chardet
from fastapi import UploadFile
from google.cloud import storage

from shared.data_model import BiosimulationsRunOutputData, BiosimulationsReportOutput, SBMLSpeciesMapping
from shared.utils import printc


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


def get_sbml_species_mapping(sbml_fp: str) -> dict:
    """

    Args:
        - sbml_fp: `str`: path to the SBML model file.

    Returns:
        Dictionary mapping of {sbml_species_names(usually the actual observable name): sbml_species_ids(ids used in the solver)}
    """
    # read file
    sbml_reader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBML(sbml_fp)
    sbml_model_object = sbml_doc.getModel()

    # parse and handle names/ids
    sbml_species_ids = []
    for spec in sbml_model_object.getListOfSpecies():
        spec_name = spec.name
        if not spec_name:
            spec.name = spec.getId()
        if not spec.name == "":
            sbml_species_ids.append(spec)
    names = list(map(lambda s: s.name, sbml_species_ids))
    species_ids = [spec.getId() for spec in sbml_species_ids]

    return dict(zip(names, species_ids))


def download_file(source_blob_path: str, out_dir: str, bucket_name: str) -> str:
    """Download any file specified in a given job_params (mongo db collection document) to out_dir. The file is assumed to originate from bucket_name.

        Returns:
            filepath (`str`) of the downloaded file.
    """
    source_blob_name = source_blob_path
    local_fp = os.path.join(out_dir, source_blob_name.split('/')[-1])
    download_blob(bucket_name=bucket_name, source_blob_name=source_blob_name, destination_file_name=local_fp)
    return local_fp


def read_uploaded_file(bucket_name, source_blob_name, destination_file_name):
    download_blob(bucket_name, source_blob_name, destination_file_name)

    with open(destination_file_name, 'r') as f:
        return f.read()


async def _save_uploaded_file(uploaded_file: UploadFile | str, save_dest: str) -> str:
    """Write `fastapi.UploadFile` instance passed by api gateway user to `save_dest`."""
    if isinstance(uploaded_file, UploadFile):
        filename = uploaded_file.filename
    else:
        filename = uploaded_file

    file_path = os.path.join(save_dest, filename)
    # case: is a fastapi upload
    if isinstance(uploaded_file, UploadFile):
        with open(file_path, 'wb') as file:
            contents = await uploaded_file.read()
            file.write(contents)
    # case: is a string
    else:
        with open(filename, 'r') as fp:
            contents = fp.read()
        with open(file_path, 'w') as f:
            f.write(contents)

    return file_path


def make_dir(fp: str, mk_temp: bool = False) -> str | None:
    if mk_temp:
        return mkdtemp()
    else:
        if not os.path.exists(fp):
            os.mkdir(fp)


def detect_encoding(file_path: os.PathLike[str]) -> dict:
    with open(file_path, 'rb') as f:
        raw_data = f.read()
        result = chardet.detect(raw_data)
        return result


def fix_non_ascii_characters(file_path: os.PathLike[str], output_file: os.PathLike[str]) -> None:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    non_ascii_chars = set(re.findall(r'[^\x00-\x7F]', content))
    replacements = {}
    for char in non_ascii_chars:
        ascii_name = unicodedata.name(char, None)
        if ascii_name:
            ascii_equivalent = ascii_name.lower().replace(" ", "_")
            replacements[char] = ascii_equivalent
        else:
            replacements[char] = ""
    for original, replacement in replacements.items():
        content = content.replace(original, replacement)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Fixed file saved to {output_file}")


def normalize_smoldyn_output_path_in_root(root_fp) -> str | None:
    new_path = None
    for root, dirs, files in os.walk(root_fp):
        for filename in files:
            if filename.endswith('out.txt'):
                original_path = os.path.join(root, filename)
                new_path = os.path.join(root, 'modelout.txt')
                os.rename(original_path, new_path)

    return new_path


def format_smoldyn_configuration(filename: str) -> None:
    config = read_smoldyn_simulation_configuration(filename)
    disable_smoldyn_graphics_in_simulation_configuration(configuration=config)
    return write_smoldyn_simulation_configuration(configuration=config, filename=filename)


def read_smoldyn_simulation_configuration(filename: str) -> List[str]:
    ''' Read a configuration for a Smoldyn simulation

    Args:
        filename (:obj:`str`): path to model file

    Returns:
        :obj:`list` of :obj:`str`: simulation configuration
    '''
    with open(filename, 'r') as file:
        return [line.strip('\n') for line in file]


def write_smoldyn_simulation_configuration(configuration: List[str], filename: str):
    ''' Write a configuration for Smoldyn simulation to a file

    Args:
        configuration
        filename (:obj:`str`): path to save configuration
    '''
    with open(filename, 'w') as file:
        for line in configuration:
            file.write(line)
            file.write('\n')


def disable_smoldyn_graphics_in_simulation_configuration(configuration: List[str]):
    ''' Turn off graphics in the configuration of a Smoldyn simulation

    Args:
        configuration (:obj:`list` of :obj:`str`): simulation configuration
    '''
    for i_line, line in enumerate(configuration):
        if line.startswith('graphics '):
            configuration[i_line] = re.sub(r'^graphics +[a-z_]+', 'graphics none', line)
