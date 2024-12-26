import os
import logging
from tempfile import mkdtemp
from abc import ABC, abstractmethod
from typing import Any

import libsbml
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from process_bigraph import Process, Step

from biosimulators_processes import CORE
from biosimulators_processes.io import unpack_omex_archive, get_archive_model_filepath, get_sedml_time_config
from biosimulators_processes.data_model.sed_data_model import UTC_CONFIG_TYPE
from biosimulators_processes.helpers import calc_duration, calc_num_steps, calc_step_size, plot_utc_outputs, check_ode_kisao_term


class SbmlUniformTimeCourse(Step):
    """ABC for UTC process declarations and simulations."""
    config_schema = UTC_CONFIG_TYPE

    def __init__(self,
                 config=None,
                 core=CORE,
                 time_config: dict = None,
                 model_source: str = None,
                 sed_model_config: dict = None):

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
        model_config = self.config['model']
        model_fp = model_config.get('model_source')
        assert model_fp is not None and '/' in model_fp, 'You must pass a valid path to an SBML model file.'

        self.simulator = self._load_simulator(model_fp)

        # set time config and model with time config
        utc_config = self.config.get('time_config')
        assert utc_config, \
            "For now you must manually pass time_config: {duration: , num_steps: , step_size: , } in the config."
        self.step_size = utc_config.get('step_size')
        self.duration = utc_config.get('duration')
        self.num_steps = utc_config.get('num_steps')
        self.initial_time = utc_config.get('initial_time', 0)
        self.output_start_time = utc_config.get('output_start_time', 0)
        if len(list(utc_config.keys())) < 3:
            self._set_time_params()

        self.species_context_key = f'floating_species'
        self.use_counts = self.config['species_context'].lower() == 'counts'

        self.floating_species_list = self._get_floating_species()
        self.model_parameters_list = self._get_model_parameters()
        self.reaction_list = self._get_reactions()
        self.t = np.linspace(self.output_start_time - 1, self.duration, self.num_steps)

        sbml_reader = libsbml.SBMLReader()
        sbml_doc = sbml_reader.readSBML(model_fp)
        self.sbml_model: libsbml.Model = sbml_doc.getModel()
        self.sbml_species_ids = [spec for spec in self.sbml_model.getListOfSpecies()]
        self.sbml_species_mapping = dict(zip(
            list(map(lambda s: s.name, self.sbml_species_ids)),
            [spec.getId() for spec in self.sbml_species_ids],
        ))

        self._results = {}
        self.output_keys = [list(self.sbml_species_mapping.keys())[i] for i, spec_id in enumerate(self.floating_species_list)]

    @staticmethod
    def _get_sedml_time_params(omex_path: str):
        sedml_fp = None
        for f in os.listdir(omex_path):
            if f.endswith('.sedml'):
                sedml_fp = os.path.join(omex_path, f)

        assert sedml_fp is not None, 'Your OMEX archive must contain a valid SEDML file.'
        # sedml_fp = os.path.join(omex_path, 'simulation.sedml')
        sedml_utc_config = get_sedml_time_config(sedml_fp)

        def convert(x): return int(x) if isinstance(x, float) else int(float(x))

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

    def _set_time_params(self):
        if self.step_size and self.num_steps:
            self.duration = calc_duration(self.num_steps, self.step_size)
        elif self.step_size and self.duration:
            self.num_steps = calc_num_steps(self.duration, self.step_size)
        else:
            self.step_size = calc_step_size(self.duration, self.num_steps)

    @abstractmethod
    def _load_simulator(self, model_fp: str, **kwargs):
        pass

    @abstractmethod
    def _get_floating_species(self) -> list[str]:
        pass

    @abstractmethod
    def _get_model_parameters(self) -> list[str]:
        pass

    @abstractmethod
    def _get_reactions(self) -> list[str]:
        pass

    @abstractmethod
    def _get_initial_state_params(self):
        """Initial params include: time, floating_species, model_parameters, and reactions."""
        pass

    @abstractmethod
    def _generate_results(self, inputs=None):
        pass

    def initial_state(self):
        return self._get_initial_state_params()

    def inputs(self):
        # dependent on species context set in self.config
        model_params_type = {
            param_id: {
                '_type': 'float',
                '_apply': 'set'}
            for param_id in self.model_parameters_list
        }

        return {
            'time': 'list[float]',
            self.species_context_key: 'tree[float]',  # floating_species_type,
            'model_parameters': model_params_type,
            'reactions': 'list[string]'}

    def outputs(self):
        return {
            'time': 'list[float]',
            self.species_context_key: 'tree[float]'}  # floating_species_type}

    def update(self, inputs=None) -> dict[str, dict[str, list[Any]] | np.ndarray[Any, np.dtype[Any]] | np.ndarray]:
        """Public method which adheres to the process bigraph interface"""
        results = self._generate_results(inputs)
        self._results = results.copy()
        return results

    def plot_results(self, simulator_name: str):
        """Default class method for plotting. May be (and most likely will be) overridden by simulator-specific plotting methods.
            Plot ODE simulation observables with Seaborn.

        """
        plt.figure(figsize=(20, 8))
        for spec_id, spec_output in self._results['floating_species'].items():
            sns.lineplot(x=self._results['time'], y=spec_output, label=spec_id)
        plt.legend()
        plt.grid(True)
        plt.title(f"Species concentrations over time with {simulator_name}")
        return plt.show()
        # return plot_utc_outputs(
            # simulator=simulator_name,
            # data=self._results,
            # t=np.append(self.t, self.t[-1] + self.step_size))

    def _flush_results(self):
        return self._results.clear()
