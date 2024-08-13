import os
from typing import Union 

import h5py
import libsbml
from biosimulators_utils.combine.io import CombineArchiveReader

from data_model import BiosimulationsRunOutputData, BiosimulationsReportOutput


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