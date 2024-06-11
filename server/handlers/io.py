import os.path
import tempfile

import h5py
import libsbml
from biosimulators_utils.combine.io import CombineArchiveReader
from biosimulator_processes.data_model.service_data_model import BiosimulationsRunOutputData, BiosimulationsReportOutput


async def get_sbml_species_names(fp: str) -> list[str]:
    sbml_reader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBML(fp)
    model: libsbml.Model = sbml_doc.getModel()
    return [s.getName() for s in model.getListOfSpecies()]


async def unpack_omex(archive_fp: str, save_dir: str):
    return CombineArchiveReader().run(archive_fp, save_dir)


async def get_sbml_model_file_from_archive(archive_fp: str, save_dir: str):
    arch = await unpack_omex(archive_fp, save_dir)
    for content in arch.contents:
        loc = content.location
        if loc.endswith('.xml') and 'manifest' not in loc.lower():
            return os.path.join(save_dir, loc)


def save_omex_archive(contents: bytes, save_dir: str):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.omex') as temp_file:
        temp_file.write(contents)
        archive_path = temp_file.name

    return {'source': archive_path, 'archive': unpack_omex(archive_path, save_dir), 'save_dir': save_dir}


async def make_dir(fp: str):
    if not os.path.exists(fp):
        os.mkdir(fp)


async def read_report_outputs(report_file_path) -> BiosimulationsRunOutputData:
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

