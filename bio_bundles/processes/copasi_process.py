from typing import Dict, Union, Optional, List, Any
from datetime import datetime
import json

import numpy as np
from COPASI import CDataModel
from numpy import ndarray, dtype, array, append as npAppend
from pandas import DataFrame
from basico import (
    load_model,
    get_species,
    get_parameters,
    get_reactions,
    set_species,
    run_time_course_with_output,
    run_time_course,
    get_compartments,
    new_model,
    set_reaction_parameters,
    add_reaction,
    set_reaction,
    set_parameters,
    add_parameter
)

from biosimulators_processes.helpers import fetch_biomodel, plot_utc_outputs
from biosimulators_processes import CORE
from biosimulators_processes.data_model.sed_data_model import UTC_CONFIG_TYPE
from biosimulators_processes.processes.utc_process import SbmlUniformTimeCourse
from biosimulators_processes.processes.sed_process import SedProcess


class UtcCopasi(SbmlUniformTimeCourse):
    config_schema = UTC_CONFIG_TYPE
    config_schema['method'] = {
        '_type': 'string',
        '_default': 'deterministic'
    }

    def __init__(self,
                 config=None,
                 core=CORE,
                 time_config: dict = None,
                 model_source: str = None,
                 sed_model_config: dict = None):
        super().__init__(config, core, time_config, model_source, sed_model_config)
        self.sbml_species_mapping = get_species(model=self.simulator)[['sbml_id']].to_dict()['sbml_id']
        self.floating_species_list = list(self.sbml_species_mapping.values())
        species_data = get_species(model=self.simulator)
        self.floating_species_initial = species_data.particle_number.tolist() \
            if self.use_counts else species_data.concentration.tolist()
        self.basico_species_ids = list(self.sbml_species_mapping.keys())

        self.model_changes = self.config['model'].get('model_changes')
        # self._set_reaction_changes()
        # self._set_species_changes()
        # self._set_global_param_changes()

        model_parameters = get_parameters(model=self.simulator)
        self.model_parameters_list = model_parameters.index.tolist() \
            if isinstance(model_parameters, DataFrame) else []
        self.model_parameters_values = model_parameters.initial_value.tolist() \
            if isinstance(model_parameters, DataFrame) else []

        reactions = get_reactions(model=self.simulator)
        self.reaction_list = reactions.index.tolist() if reactions is not None else []

        self.compartments_list = get_compartments(model=self.simulator).index.tolist()

        # ----SOLVER: Get the solver (defaults to deterministic)
        self.method = self.config['method']
        self._tc = None

    def _get_initial_state_params(self):
        # keep in mind that a valid simulation may not have global parameters
        model_parameters_dict = dict(
            zip(self.model_parameters_list, self.model_parameters_values))

        floating_species_dict = dict(
            zip(self.floating_species_list, self.floating_species_initial))

        return {
            'time': [0.0],
            'model_parameters': model_parameters_dict,
            self.species_context_key: floating_species_dict,
            'reactions': self.reaction_list
        }

    def plot_results(self):
        return super().plot_results(simulator_name='COPASI')

    def _load_simulator(self, model_fp: str, **kwargs):
        return load_model(model_fp)

    def _get_reactions(self) -> list[str]:
        reactions = get_reactions(model=self.simulator)
        return reactions.index.tolist() if reactions is not None else []

    def _get_floating_species(self) -> list[str]:
        species_data = get_species(model=self.simulator)
        return species_data.index.tolist()

    def _get_model_parameters(self) -> list[str]:
        model_parameters = get_parameters(model=self.simulator)
        return model_parameters.index.tolist() \
            if isinstance(model_parameters, DataFrame) else []

    def _generate_results(self, inputs=None):
        x = inputs or self.initial_state()
        # get the copasi-familiar names
        reported_outputs = [k for k in self.sbml_species_mapping.keys()]
        reported_outputs.append('Time')

        t = np.linspace(self.output_start_time - 1, self.duration + 1, self.num_steps + 1)
        # run specified output range
        self._tc = run_time_course_with_output(
            # start_time=self.initial_time,
            start_time=self.output_start_time,
            duration=t[-1],
            values=t,
            model=self.simulator,
            update_model=True,
            output_selection=reported_outputs,
            use_numbers=True)

        tc = self._tc.to_dict()
        results = {'time': array(list(tc['Time'].values())), 'floating_species': {}}
        keys = [list(self.sbml_species_mapping.keys())[i] for i, spec_id in enumerate(self.floating_species_list)]
        for i, name in enumerate(self.floating_species_list):
            results['floating_species'][self.output_keys[i]] = array(list(tc.get(self.basico_species_ids[i]).values()))  # [self.output_start_time:self.duration]
        return results

    def _set_reaction_changes(self):
        # ----REACTIONS: set reactions
        existing_reactions = get_reactions(model=self.simulator)
        existing_reaction_names = existing_reactions.index.tolist() if existing_reactions is not None else []
        reaction_changes = self.model_changes.get('reaction_changes', [])
        if reaction_changes:
            for reaction_change in reaction_changes:
                reaction_name: str = reaction_change['reaction_name']
                param_changes: list[dict[str, float]] = reaction_change['parameter_changes']
                scheme_change: str = reaction_change.get('reaction_scheme')
                # handle changes to existing reactions
                if param_changes:
                    for param_name, param_change_val in param_changes:
                        set_reaction_parameters(param_name, value=param_change_val, model=self.simulator)
                if scheme_change:
                    set_reaction(name=reaction_name, scheme=scheme_change, model=self.simulator)
                # handle new reactions
                if reaction_name not in existing_reaction_names and scheme_change:
                    add_reaction(reaction_name, scheme_change, model=self.simulator)

    def _set_species_changes(self):
        # ----SPECS: set species changes
        species_changes = self.model_changes.get('species_changes', [])
        if species_changes:
            for species_change in species_changes:
                if isinstance(species_change, dict):
                    species_name = species_change.pop('name')
                    changes_to_apply = {}
                    for spec_param_type, spec_param_value in species_change.items():
                        if spec_param_value:
                            changes_to_apply[spec_param_type] = spec_param_value
                    set_species(**changes_to_apply, model=self.simulator)

    def _set_global_param_changes(self):
        # ----GLOBAL PARAMS: set global parameter changes
        global_parameter_changes = self.model_changes.get('global_parameter_changes', [])
        if global_parameter_changes:
            for param_change in global_parameter_changes:
                param_name = param_change.pop('name')
                for param_type, param_value in param_change.items():
                    if not param_value:
                        param_change.pop(param_type)
                    # handle changes to existing params
                    set_parameters(name=param_name, **param_change, model=self.simulator)
                    # set new params
                    global_params = get_parameters(model=self.simulator)
                    if global_params:
                        existing_global_parameters = global_params.index
                        if param_name not in existing_global_parameters:
                            assert param_change.get('initial_concentration') is not None, "You must pass an initial_concentration value if adding a new global parameter."
                            add_parameter(name=param_name, **param_change, model=self.simulator)


class CopasiProcess(SedProcess):
    config_schema = UTC_CONFIG_TYPE

    def __init__(self,
                 config: Dict[str, Union[str, Dict[str, str], Dict[str, Optional[Dict[str, str]]], Optional[Dict[str, str]]]] = None,
                 core: Dict = CORE):
        super().__init__(config, core)

        # insert copasi process model config
        model_source = self.config['model'].get('model_source') or self.config.get('sbml_fp')
        assert model_source is not None, 'You must specify a model source of either a valid biomodel id or model filepath.'
        model_changes = self.config['model'].get('model_changes', {})
        self.model_changes = {} if model_changes is None else model_changes

        # Option A:
        if '/' in model_source:
            self.copasi_model_object = load_model(model_source)
            print('found a filepath')

        # Option B:
        elif 'BIO' in model_source:
            self.copasi_model_object = fetch_biomodel(model_id=model_source)
            print('found a biomodel id')

        # Option C:
        else:
            if not self.model_changes:
                raise AttributeError(
                    """You must pass a source of model changes specifying params, reactions, 
                        species or all three if starting from an empty model.""")
            model_units = self.config['model'].get('model_units', {})
            self.copasi_model_object = new_model(
                name='CopasiProcess TimeCourseModel',
                **model_units)

        # handle context of species output
        context_type = self.config['species_context']
        self.species_context_key = f'floating_species_{context_type}'
        self.use_counts = 'concentrations' in context_type

        # Get a list of reactions
        self._set_reaction_changes()
        reactions = get_reactions(model=self.copasi_model_object)
        self.reaction_list = reactions.index.tolist() if reactions is not None else []
        # if not self.reaction_list:
            # raise AttributeError('No reactions could be parsed from this model. Your model must contain reactions to run.')

        # Get the species (floating only)  TODO: add boundary species
        self._set_species_changes()
        species_data = get_species(model=self.copasi_model_object)
        self.floating_species_list = species_data.index.tolist()
        self.floating_species_initial = species_data.particle_number.tolist() \
            if self.use_counts else species_data.concentration.tolist()

        # Get the list of parameters and their values (it is possible to run a model without any parameters)
        self._set_global_param_changes()
        model_parameters = get_parameters(model=self.copasi_model_object)
        self.model_parameters_list = model_parameters.index.tolist() \
            if isinstance(model_parameters, DataFrame) else []
        self.model_parameters_values = model_parameters.initial_value.tolist() \
            if isinstance(model_parameters, DataFrame) else []

        # Get a list of compartments
        self.compartments_list = get_compartments(model=self.copasi_model_object).index.tolist()

        # ----SOLVER: Get the solver (defaults to deterministic)
        self.method = self.config['method']

    def initial_state(self):
        # keep in mind that a valid simulation may not have global parameters
        model_parameters_dict = dict(
            zip(self.model_parameters_list, self.model_parameters_values))

        floating_species_dict = dict(
            zip(self.floating_species_list, self.floating_species_initial))

        return {
            'time': 0.0,
            'model_parameters': model_parameters_dict,
            self.species_context_key: floating_species_dict,
        }

    def inputs(self):
        # dependent on species context set in self.config
        floating_species_type = {
            species_id: {
                '_type': 'float',
                '_apply': 'set'}
            for species_id in self.floating_species_list
        }

        model_params_type = {
            param_id: {
                '_type': 'float',
                '_apply': 'set'}
            for param_id in self.model_parameters_list
        }

        reactions_type = {
            reaction_id: 'float'
            for reaction_id in self.reaction_list
        }

        return {
            'time': 'float',
            self.species_context_key: floating_species_type,
            'model_parameters': model_params_type,
            'reactions': reactions_type}

    def outputs(self):
        floating_species_type = {
            species_id: {
                '_type': 'float',
                '_apply': 'set'}
            for species_id in self.floating_species_list
        }
        return {
            'time': 'float',
            self.species_context_key: floating_species_type}

    def update(self, inputs, interval):
        # set copasi values according to what is passed in states for concentrations
        for cat_id, value in inputs[self.species_context_key].items():
            set_type = 'particle_number' if 'counts' in self.species_context_key else 'concentration'
            species_config = {
                'name': cat_id,
                'model': self.copasi_model_object,
                set_type: value}
            set_species(**species_config)

        # run model for "interval" length; we only want the state at the end
        timecourse = run_time_course(
            start_time=inputs['time'],
            duration=interval,
            update_model=True,
            model=self.copasi_model_object,
            method=self.method)

        # extract end values of concentrations from the model and set them in results
        results = {'time': interval}
        if self.use_counts:
            results[self.species_context_key] = {
                mol_id: float(get_species(
                    name=mol_id,
                    exact=True,
                    model=self.copasi_model_object
                ).particle_number[0])
                for mol_id in self.floating_species_list}
        else:
            results[self.species_context_key] = {
                mol_id: float(get_species(
                    name=mol_id,
                    exact=True,
                    model=self.copasi_model_object
                ).concentration[0])
                for mol_id in self.floating_species_list}

        return results

    def _set_reaction_changes(self):
        # ----REACTIONS: set reactions
        existing_reactions = get_reactions(model=self.copasi_model_object)
        existing_reaction_names = existing_reactions.index.tolist() if existing_reactions is not None else []
        reaction_changes = self.model_changes.get('reaction_changes', [])
        if reaction_changes:
            for reaction_change in reaction_changes:
                reaction_name: str = reaction_change['reaction_name']
                param_changes: list[dict[str, float]] = reaction_change['parameter_changes']
                scheme_change: str = reaction_change.get('reaction_scheme')
                # handle changes to existing reactions
                if param_changes:
                    for param_name, param_change_val in param_changes:
                        set_reaction_parameters(param_name, value=param_change_val, model=self.copasi_model_object)
                if scheme_change:
                    set_reaction(name=reaction_name, scheme=scheme_change, model=self.copasi_model_object)
                # handle new reactions
                if reaction_name not in existing_reaction_names and scheme_change:
                    add_reaction(reaction_name, scheme_change, model=self.copasi_model_object)

    def _set_species_changes(self):
        # ----SPECS: set species changes
        species_changes = self.model_changes.get('species_changes', [])
        if species_changes:
            for species_change in species_changes:
                if isinstance(species_change, dict):
                    species_name = species_change.pop('name')
                    changes_to_apply = {}
                    for spec_param_type, spec_param_value in species_change.items():
                        if spec_param_value:
                            changes_to_apply[spec_param_type] = spec_param_value
                    set_species(**changes_to_apply, model=self.copasi_model_object)

    def _set_global_param_changes(self):
        # ----GLOBAL PARAMS: set global parameter changes
        global_parameter_changes = self.model_changes.get('global_parameter_changes', [])
        if global_parameter_changes:
            for param_change in global_parameter_changes:
                param_name = param_change.pop('name')
                for param_type, param_value in param_change.items():
                    if not param_value:
                        param_change.pop(param_type)
                    # handle changes to existing params
                    set_parameters(name=param_name, **param_change, model=self.copasi_model_object)
                    # set new params
                    global_params = get_parameters(model=self.copasi_model_object)
                    if global_params:
                        existing_global_parameters = global_params.index
                        if param_name not in existing_global_parameters:
                            assert param_change.get('initial_concentration') is not None, "You must pass an initial_concentration value if adding a new global parameter."
                            add_parameter(name=param_name, **param_change, model=self.copasi_model_object)
