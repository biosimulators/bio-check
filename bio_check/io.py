import os.path

import h5py
from fastapi import UploadFile

from biosimulator_processes.data_model.service_data_model import BiosimulationsRunOutputData, BiosimulationsReportOutput


async def save_uploaded_file(uploaded_file: UploadFile, save_dest: str) -> str:
    # TODO: replace this with s3 and use save_dest
    file_path = os.path.join(save_dest, uploaded_file.filename)
    with open(file_path, 'wb') as file:
        contents = await uploaded_file.read()
        file.write(contents)
    return file_path


def make_dir(fp: str):
    if not os.path.exists(fp):
        os.mkdir(fp)


async def read_report_outputs_async(report_file_path: str) -> BiosimulationsRunOutputData:
    return read_report_outputs(report_file_path)


def read_report_outputs(report_file_path) -> BiosimulationsRunOutputData:
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
            print(f"Group '{group_path}' not found in the file.")

