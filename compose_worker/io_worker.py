import os
import re
from tempfile import mkdtemp
from typing import Union, List

import h5py
import libsbml
from biosimulators_utils.combine.io import CombineArchiveReader

from fastapi import UploadFile
from google.cloud import storage

from data_model import BiosimulationsRunOutputData, BiosimulationsReportOutput


def download_file(source_blob_path: str, out_dir: str, bucket_name: str) -> str:
    """Download any file specified in a given job_params (mongo db collection document) to out_dir. The file is assumed to originate from bucket_name.

        Returns:
            filepath (`str`) of the downloaded file.
    """
    source_blob_name = source_blob_path
    local_fp = os.path.join(out_dir, source_blob_name.split('/')[-1])
    download_blob(bucket_name=bucket_name, source_blob_name=source_blob_name, destination_file_name=local_fp)
    return local_fp


async def write_uploaded_file(job_id: str, bucket_name: str, uploaded_file: UploadFile | str, extension: str, save_dest=None) -> str:
    # bucket params
    upload_prefix = f"file_uploads/{job_id}/"
    bucket_prefix = f"gs://{bucket_name}/" + upload_prefix

    save_dest = save_dest or mkdtemp()
    fp = await save_uploaded_file(uploaded_file, save_dest)  # save uploaded file to ephemeral store

    # Save uploaded omex file to Google Cloud Storage
    purpose = 'uploaded_file'
    properly_formatted_file = check_upload_file_extension(uploaded_file, purpose, extension)
    if not properly_formatted_file:
        raise ValueError(f"Files for {purpose} must be passed in {extension} format.")

    blob_dest = upload_prefix + fp.split("/")[-1]
    print(f'Blob destination for {blob_dest}')
    upload_blob(bucket_name=bucket_name, source_file_name=fp, destination_blob_name=blob_dest)
    print(f'blob_dest: {blob_dest}')
    return blob_dest


def check_upload_file_extension(file: UploadFile | str, purpose: str, ext: str) -> bool:
    fname = file.filename if isinstance(file, UploadFile) else file
    return False if not fname.endswith(ext) else True


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

    blob.upload_from_filename(source_file_name)  # if_generation_match=generation_match_precondition)

    return {
        'message': f"File {source_file_name} uploaded to {destination_blob_name}."
    }


def read_uploaded_file(bucket_name, source_blob_name, destination_file_name):
    download_blob(bucket_name, source_blob_name, destination_file_name)

    with open(destination_file_name, 'r') as f:
        return f.read()


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


async def save_uploaded_file(uploaded_file: UploadFile | str, save_dest: str) -> str:
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


def make_dir(fp: str):
    if not os.path.exists(fp):
        os.mkdir(fp)


def get_sbml_species_names(fp: str) -> list[str]:
    sbml_reader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBML(fp)
    model: libsbml.Model = sbml_doc.getModel()
    return [s.getName() for s in model.getListOfSpecies()]


def unpack_omex(archive_fp: str, save_dir: str):
    return CombineArchiveReader().run(archive_fp, save_dir)


def get_sbml_model_file_from_archive(archive_fp: str, save_dir: str):
    arch = unpack_omex(archive_fp, save_dir)
    for content in arch.contents:
        loc = content.location
        if loc.endswith('.xml') and 'manifest' not in loc.lower():
            return os.path.join(save_dir, loc)


async def read_report_outputs_async(report_file_path: str) -> Union[BiosimulationsRunOutputData, str]:
    return read_report_outputs(report_file_path)


def read_report_outputs(report_file_path) -> Union[BiosimulationsRunOutputData, str]:
    """Read the outputs from all species in the given report file from biosimulations output.
        Args:
            report_file_path (str): The path to the simulation.sedml/report.h5 HDF5 file.
    """
    # TODO: implement auto gen from run id here.
    outputs = []
    with h5py.File(report_file_path, 'r') as f:
        k = list(f.keys())
        group_path = k[0] + '/report'
        if group_path in f:
            group = f[group_path]
            dataset_labels = group.attrs['sedmlDataSetLabels']
            for label in dataset_labels:
                dataset_index = list(dataset_labels).index(label)
                data = group[()]
                specific_data = data[dataset_index]
                output = BiosimulationsReportOutput(dataset_label=label, data=specific_data)
                outputs.append(output)
            return BiosimulationsRunOutputData(report_path=report_file_path, data=outputs)
        else:
            return f"Group '{group_path}' not found in the file."


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

