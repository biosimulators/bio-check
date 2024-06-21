import os
from typing import Dict

import h5py
from fastapi import UploadFile


# TODO: Configure this with an S3-like and import.
async def _save_uploaded_file(file: UploadFile):
    out_file_path = f"/path/to/shared/storage/{file.filename}"
    with open(out_file_path, 'wb') as out_file:
        content = await file.read()
        out_file.write(content)
    return out_file_path


async def save_uploaded_file(uploaded_file: UploadFile, save_dest: str) -> str:
    # TODO: replace this with s3 and use save_dest
    file_path = os.path.join(save_dest, uploaded_file.filename)
    with open(file_path, 'wb') as file:
        contents = await uploaded_file.read()
        file.write(contents)
    return file_path


async def read_report_outputs(report_file_path) -> Dict:
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
                output = {'dataset_label': label, 'data': specific_data}
                outputs.append(output)
            return {'report_path': report_file_path, 'data': outputs}
        else:
            print(f"Group '{group_path}' not found in the file.")