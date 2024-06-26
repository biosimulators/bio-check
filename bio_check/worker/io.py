import os

import libsbml
from biosimulators_utils.combine.io import CombineArchiveReader


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