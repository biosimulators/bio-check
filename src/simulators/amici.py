import logging

from amici import amici, sbml_import, import_model_module


def load_amici_model(sbml_fp: str, model_output_dir=None, **config) -> amici.Model:
    """compile sbml to amici.

        Args:
            sbml_fp (`str`): path to sbml file.
            model_output_dir (`str, optional`): path to which the compiled amici model can be stored.
            **config (`kwargs`): each pertaining to amici model config: `observables`, `constant_parameters`,
                `sigmas`. See Amici documentation for more details.

        Returns:
            amici.Model: compiled amici model.
    """
    model_id = sbml_fp.split('/')[-1].replace('.', '_').split('_')[0]
    sbml_importer = sbml_import.SbmlImporter(sbml_fp)
    sbml_importer.sbml2amici(
        model_name=model_id,
        output_dir=model_output_dir,
        verbose=logging.INFO,
        observables=config.get('observables'),
        constant_parameters=config.get('constant_parameters'),
        sigmas=config.get('sigmas'))
    model_module = import_model_module(model_id, model_output_dir)

    return model_module.getModel()
