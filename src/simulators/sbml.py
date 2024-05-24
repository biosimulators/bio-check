import libsbml


def load_sbml_model(sbml_fp: str) -> libsbml.Model:
    sbml_reader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBML(sbml_fp)
    return sbml_doc.getModel()
