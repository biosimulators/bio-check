import os
import logging
from tempfile import mkdtemp

import libsbml
import numpy as np
import seaborn as sns
from amici import amici, SbmlImporter, import_model_module, Model, runAmiciSimulation
from amici.sbml_import import get_species_initial
from matplotlib import pyplot as plt
from process_bigraph import Process, Step

from biosimulators_processes import CORE
from biosimulators_processes.io import unpack_omex_archive, get_archive_model_filepath, get_sedml_time_config
from biosimulators_processes.data_model.sed_data_model import UTC_CONFIG_TYPE
from biosimulators_processes.processes.utc_process import SbmlUniformTimeCourse
from biosimulators_processes.helpers import calc_duration, calc_num_steps, calc_step_size, check_ode_kisao_term


class UtcAmici(Step):
    """
       Parameters:
            config:`Dict`: dict keys include:
                model: SED Model Spec.
                species_context: Context by which to measure species outputs (defaults to concentrations).
                model_output_dir: Dirpath in which to save the AMICI model output.
                observables: for example:
                    observables = {
                        "observable_x1": {"name": "", "formula": "x1"},
                        "observable_x2": {"name": "", "formula": "x2"},
                        "observable_x3": {"name": "", "formula": "x3"},
                        "observable_x1_scaled": {"name": "", "formula": "scaling_x1 * x1"},
                        "observable_x2_offsetted": {"name": "", "formula": "offset_x2 + x2"},
                        "observable_x1withsigma": {"name": "", "formula": "x1"},
                    }
                constant_parameters: for example:
                    constant_parameters = ["k0"]
                sigmas: for example:
                    sigmas = {"observable_x1withsigma": "observable_x1withsigma_sigma"}

    """
    config_schema = UTC_CONFIG_TYPE

    # AMICI-specific fields
    config_schema['model_output_dir'] = {
        '_default': mkdtemp(),
        '_type': 'string'
    }
    config_schema['observables'] = 'maybe[tree[string]]'
    config_schema['constant_parameters'] = 'maybe[list[string]]'
    config_schema['sigmas'] = 'maybe[tree[string]]'

    def __init__(self,
                 config=None,
                 core=CORE,
                 time_config: dict = None,
                 model_source: str = None,
                 sed_model_config: dict = None):
        self.slice = ()
        # A. no config but either an omex file/dir or sbml file path
        configuration = config or {}
        source = configuration.get('model').get('model_source')
        archive_dir_source = os.path.isdir(source)
        if not configuration and model_source:
            configuration = {'model': {'model_source': model_source}}

        # B. has a config but wishes to override TODO: fix this.
        if sed_model_config and configuration:
            configuration['model'] = sed_model_config

        # C. has a source of sbml file path as expected.
        elif source and source.endswith('.xml') and not source.lower().startswith('manifest'):
            pass

        # D. has a config passed with an archive dirpath or filepath or sbml filepath as its model source:
        else:
            omex_path = configuration.get('model').get('model_source')
            # Da: user has passed a dirpath of omex archive or the path to an unzipped archive as model source
            archive_dir = unpack_omex_archive(archive_filepath=source, working_dir=config.get('working_dir') or mkdtemp()) if not archive_dir_source else source

            # set expected model path for init
            configuration['model']['model_source'] = get_archive_model_filepath(archive_dir)

            # extract the time config from archive's sedml
            configuration['time_config'] = self._get_sedml_time_params(archive_dir)

        if time_config and not len(configuration.get('time_config', {}).keys()):
            configuration['time_config'] = time_config

        super().__init__(config=configuration, core=core)

        # reference model source and assert filepath
        model_fp = self.config['model'].get('model_source')
        assert model_fp is not None and '/' in model_fp, 'You must pass a valid path to an SBML model file.'
        model_config = self.config['model']

        # get and compile libsbml model from fp
        sbml_reader = libsbml.SBMLReader()
        sbml_doc = sbml_reader.readSBML(model_fp)
        self.sbml_model_object: libsbml.Model = sbml_doc.getModel()

        # get model args from config
        model_id = self.config['model'].get('model_id', None) \
            or model_fp.split('/')[-1].replace('.', '_').split('_')[0]

        model_output_dir = self.config['model_output_dir']

        # compile sbml to amici
        sbml_importer = SbmlImporter(model_fp)
        sbml_importer.sbml2amici(
            model_name=model_id,
            output_dir=model_output_dir,
            verbose=logging.INFO,
            observables=self.config.get('observables'),
            constant_parameters=self.config.get('constant_parameters'),
            sigmas=self.config.get('sigmas'))
        model_module = import_model_module(model_id, model_output_dir)
        self.amici_model_object: amici.Model = model_module.getModel()

        # set species context (concentrations for ODE by default)
        self.context_type = self.config['species_context']
        self.species_context_key = f'floating_species'
        self.use_counts = 'counts' in self.species_context_key

        # get species names
        self.floating_species_list = list(self.amici_model_object.getStateIds())
        self.floating_species_initial = list(self.amici_model_object.getInitialStates())
        self.sbml_species_ids = [spec for spec in self.sbml_model_object.getListOfSpecies()]
        self.sbml_species_mapping = dict(zip(
            list(map(lambda s: s.name, self.sbml_species_ids)),
            [spec.getId() for spec in self.sbml_species_ids],
        ))

        # get model parameters
        self.model_parameter_objects = self.sbml_model_object.getListOfParameters()
        self.model_parameters_list = [param.getName() for param in self.model_parameter_objects]
        self.model_parameters_values = [param.getValue() for param in self.model_parameter_objects]

        # get reactions
        self.reaction_objects = self.sbml_model_object.reactions
        self.reaction_list = [reaction.getName() for reaction in self.reaction_objects]

        # get method
        self.method = self.amici_model_object.getSolver()

        # set time config and model with time config
        utc_config = self.config.get('time_config')
        assert utc_config, \
            "For now you must manually pass time_config: {duration: , num_steps: , step_size: , } in the config."
        self.step_size = utc_config.get('step_size')
        self.duration = utc_config.get('duration')
        self.num_steps = utc_config.get('num_steps')
        self.initial_time = utc_config.get('initial_time') or 0
        self.output_start_time = utc_config.get('output_start_time')

        if len(list(utc_config.keys())) < 3:
            self._set_time_params()

        self.t = np.linspace(self.output_start_time, self.duration, self.num_steps)

        self.amici_model_object.setTimepoints(self.t)
        self._results = {}
        self.output_keys = [list(self.sbml_species_mapping.keys())[i] for i, spec_id in enumerate(self.floating_species_list)]

    def initial_state(self):
        species_initial = dict(zip(self.floating_species_list, self.floating_species_initial))
        model_params_initial = dict(zip(self.model_parameters_list, self.model_parameters_values))
        return {
            'time': [0.0],
            'floating_species': species_initial,
            'model_parameters': model_params_initial,
            'reactions': self.reaction_list
        }

    @staticmethod
    def _get_sedml_time_params(omex_path: str):
        sedml_fp = None
        for f in os.listdir(omex_path):
            if f.endswith('.sedml'):
                sedml_fp = os.path.join(omex_path, f)

        assert sedml_fp is not None, 'Your OMEX archive must contain a valid SEDML file.'
        # sedml_fp = os.path.join(omex_path, 'simulation.sedml')
        sedml_utc_config = get_sedml_time_config(sedml_fp)

        def convert(x):
            return int(x) if isinstance(x, float) else int(float(x))

        output_end = convert(sedml_utc_config['outputEndTime'])
        output_start = convert(sedml_utc_config['outputStartTime'])
        duration = output_end - output_start
        n_steps = convert(sedml_utc_config['numberOfPoints'])
        initial_time = convert(sedml_utc_config['initialTime'])
        step_size = calc_step_size(duration, n_steps)

        # check kisao id for supported algorithm/kisao ID
        specified_alg = sedml_utc_config['algorithm'].split(':')[1]
        supported_alg = check_ode_kisao_term(specified_alg)
        if not supported_alg:
            raise Exception('Algorithm specified in OMEX archive is non-deterministic and thus not supported by a Uniform Time Course implementation.')

        return {
            'duration': output_end,  # duration,
            'num_steps': n_steps,  # to account for self comparison
            'step_size': step_size,
            'output_start_time': output_start,
            'initial_time': initial_time
        }

    def plot_results(self, flush=True):
        """Plot ODE simulation observables with Seaborn."""
        plt.figure(figsize=(20, 8))
        for n in range(len(self.floating_species_list)):
            sns.lineplot(x=self._results['time'], y=list(self._results['floating_species'].values())[n])
        plt.xlabel('Time')
        plt.ylabel('Concentrations')
        plt.title('Species Concentrations over Time with AMICI')
        return self.flush_results() if flush else None

    def flush_results(self):
        return self._results.clear()

    def _set_time_params(self):
        if self.step_size and self.num_steps:
            self.duration = calc_duration(self.num_steps, self.step_size)
        elif self.step_size and self.duration:
            self.num_steps = calc_num_steps(self.duration, self.step_size)
        else:
            self.step_size = calc_step_size(self.duration, self.num_steps)

    def inputs(self):
        # dependent on species context set in self.config
        return {
            'time': 'list[float]',
            self.species_context_key: 'tree[float]',
            'model_parameters': 'tree[float]',
            'reactions': 'list[string]'}

    def outputs(self):
        return {
            'time': 'list[float]',
            self.species_context_key: 'tree[float]'}  # floating_species_type}

    def _generate_results(self, inputs=None):
        x = inputs or self.initial_state()
        if len(x.keys()):
            set_values = []
            for species_id, value in x[self.species_context_key].items():
                set_values.append(value)
            self.amici_model_object.setInitialStates(set_values)

        result_data = runAmiciSimulation(solver=self.method, model=self.amici_model_object)

        # TODO: ensure that `keys` are threadsafe.
        floating_species_results = dict(zip(
            self.output_keys,
            list(map(lambda x: result_data.by_id(f'{x}'), self.floating_species_list))))

        return {
            'time': self.t,
            self.species_context_key: floating_species_results}

    def update(self, inputs=None):
        results = self._generate_results(inputs)
        self._results = results.copy()
        return results
