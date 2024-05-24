from typing import Dict, Union, Optional, List
from datetime import datetime
import json

from process_bigraph import Process
from COPASI import CDataModel
from pandas import DataFrame
from basico import (
    load_model,
    get_species,
    get_parameters,
    get_reactions,
    set_species,
    run_time_course,
    get_compartments,
    new_model,
    set_reaction_parameters,
    add_reaction,
    set_reaction,
    set_parameters,
    add_parameter
)


from src import CORE
from src.processes.schemas import MODEL_TYPE
from src.processes.sed_process import SedProcess


class CopasiProcess(SedProcess):
    """
        Entrypoint Options:

            A. Filepath whose reference is a valid SBML model file(`str`),
            B. A valid BioModel id which will return a simulator-instance object(`str`),
            C. A specification of model configuration parameters whose values reflect those required by BasiCo. This
                specification requires the presence of reactions which inherently define species types and optionally
                addition parameters. See Basico for more details. There are two types of objects that are accepted
                for this specification:
                    - A high-level server object from `biosimulator_processes.data_model`,
                        ie: `TimeCourseModel` or `SedModel`. (Recommended for first-time users). The parameters with
                        which these dataclasses are instantiated correspond to the `config_schema` for a given
                        process implementation. In the config schema, the outermost keys could be considered
                        parameters/kwargs for a process implementation's construction. The values are all terminally
                        strings that define the parameter "type" according to bigraph-schema.
                    - A dictionary which defines the same kwargs/values as the high-level objects. See
                        `biosimulator_processes.data_model.MODEL_TYPE` for details.

        Config:
            model: see datamodel for more details.
            species_context: the context by which you measure the species data:: one of: 'concentrations', 'counts'. # TODO: map these to method inference
            method: basico timecourse setting. Defaults to 'lsoda'.
    """

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



class workingcopasiprocess(SedProcess):
    """
        Entrypoint Options:

            A. Filepath whose reference is a valid SBML model file(`str`),
            B. A valid BioModel id which will return a simulator-instance object(`str`),
            C. A specification of model configuration parameters whose values reflect those required by BasiCo. This
                specification requires the presence of reactions which inherently define species types and optionally
                addition parameters. See Basico for more details. There are two types of objects that are accepted
                for this specification:
                    - A high-level server object from `biosimulator_processes.data_model`,
                        ie: `TimeCourseModel` or `SedModel`. (Recommended for first-time users). The parameters with
                        which these dataclasses are instantiated correspond to the `config_schema` for a given
                        process implementation. In the config schema, the outermost keys could be considered
                        parameters/kwargs for a process implementation's construction. The values are all terminally
                        strings that define the parameter "type" according to bigraph-schema.
                    - A dictionary which defines the same kwargs/values as the high-level objects. See
                        `biosimulator_processes.data_model.MODEL_TYPE` for details.

        Config:
            model: see datamodel for more details.
            species_context: the context by which you measure the species data:: one of: 'concentrations', 'counts'. # TODO: map these to method inference
            method: basico timecourse setting. Defaults to 'lsoda'.
    """

    config_schema = {
        'model': MODEL_TYPE,
        'species_context': {
            '_type': 'string',
            '_default': 'concentrations'
        },
        'method': {
            '_type': 'string',
            '_default': 'deterministic'  # <-- CVODE for consistency, or should we use LSODA?
        },
        'sbml_fp': 'maybe[string]'
    }

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


class _CopasiProcess(Process):
    config_schema = {
        'model': MODEL_TYPE,
        'species_context': {
            '_type': 'string',
            '_default': 'concentrations'
        },
        'method': {
            '_type': 'string',
            '_default': 'deterministic'  # <-- CVODE for consistency, or should we use LSODA?
        }
    }

    model_changes: Dict
    species_context_key: str
    use_counts: bool
    reaction_list: List[str]
    floating_species_list: List[str]
    copasi_model_object: CDataModel
    floating_species_initial: List[float]
    model_parameters_list: List[str]
    model_parameters_values: List[float]
    compartments_list: List[str]
    method: str

    def __init__(self,
                 config: Dict[str, Union[str, Dict[str, str], Dict[str, Optional[Dict[str, str]]], Optional[Dict[str, str]]]] = None,
                 core: Dict = CORE):
        super().__init__(config, core)

    def initial_state(self):
        # insert copasi process model config
        model_source = self.config['model'].get('model_source')
        assert model_source is not None, "You must pass a model source."
        self.copasi_model_object = load_model(model_source)

        model_changes = self.config['model'].get('model_changes', {})
        self.model_changes = {} if model_changes is None else model_changes

        # handle context of species output
        context_type = self.config['species_context']
        self.species_context_key = f'floating_species_{context_type}'
        self.use_counts = 'concentrations' in context_type

        # Get a list of reactions
        # self._set_reaction_changes()
        reactions = get_reactions(model=self.copasi_model_object)
        self.reaction_list = reactions.index.tolist() if reactions is not None else []
        # if not self.reaction_list:
        # raise AttributeError('No reactions could be parsed from this model. Your model must contain reactions to run.')

        # Get the species (floating only)  TODO: add boundary species
        # self._set_species_changes()
        species_data = get_species(model=self.copasi_model_object)
        self.floating_species_list = species_data.index.tolist()
        self.floating_species_initial = species_data.particle_number.tolist() \
            if self.use_counts else species_data.concentration.tolist()

        # Get the list of parameters and their values (it is possible to run a model without any parameters)
        # self._set_global_param_changes()
        model_parameters = get_parameters(model=self.copasi_model_object)
        self.model_parameters_list = model_parameters.index.tolist() \
            if isinstance(model_parameters, DataFrame) else []
        self.model_parameters_values = model_parameters.initial_value.tolist() \
            if isinstance(model_parameters, DataFrame) else []

        # Get a list of compartments
        self.compartments_list = get_compartments(model=self.copasi_model_object).index.tolist()

        # ----SOLVER: Get the solver (defaults to deterministic)
        self.method = self.config['method']
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
            'model_source': 'string',
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
