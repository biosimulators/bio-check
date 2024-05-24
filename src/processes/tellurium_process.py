"""
Tellurium Process
"""


import numpy as np
import tellurium as te
from tellurium.roadrunner.extended_roadrunner import ExtendedRoadRunner
from process_bigraph import Process, Composite, pf, Step

from src.processes import CORE
from src.processes.schemas import SED_MODEL_TYPE


class TelluriumStep(Step):

    config_schema = {
        'sbml_filepath': 'string',
        'num_points': 'integer',
        'duration': 'integer',
        'start_time': {
            '_type': 'integer',
            '_default': 0
        }
    }

    def __init__(self, sbml_filepath: str, num_points: int, duration: int, config=None, core=None):
        config = config or {
            'sbml_filepath': sbml_filepath,
            'num_points': num_points,
            'duration': duration
        }
        super().__init__(config=config, core=core)

        self.simulator: ExtendedRoadRunner = te.loadSBMLModel(self.config['sbml_filepath'])

        self.input_ports = [
            'floating_species',
            'boundary_species',
            'model_parameters'
        ]

        self.output_ports = [
            'floating_species',
        ]

        # Get the species (floating and boundary)
        self.floating_species_list = self.simulator.getFloatingSpeciesIds()
        self.boundary_species_list = self.simulator.getBoundarySpeciesIds()
        self.floating_species_initial = self.simulator.getFloatingSpeciesConcentrations()
        self.boundary_species_initial = self.simulator.getBoundarySpeciesConcentrations()

        # Get the list of parameters and their values
        self.model_parameters_list = self.simulator.getGlobalParameterIds()
        self.model_parameter_values = self.simulator.getGlobalParameterValues()

        # Get a list of reactions
        self.reaction_list = self.simulator.getReactionIds()

        # Get num points
        self.start_time = float(self.config['start_time'])
        self.duration = float(self.config['duration'])
        self.num_points = self.config['num_points']

    # TODO -- is initial state even working for steps?
    def initial_state(self, config=None):
        return {
            'inputs': {
                'start_time': self.start_time,
                'duration': self.duration
            },
        }

    def inputs(self):
        return {
            'start_time': 'float',
            'duration': 'float',
        }

    def outputs(self):
        return {
            'results': {
                '_type': 'numpy_array',
                '_apply': 'set'
            }  # This is a roadrunner._roadrunner.NamedArray
        }

    def update(self, inputs=None):
        results = self.simulator.simulate(
            self.start_time,
            self.duration,
            self.num_points
            # inputs['time'],
            # inputs['run_time'],
            # 10
        )  # TODO -- adjust the number of saves teps
        return {
            'results': results}


# Tellurium Process
class TelluriumProcess(Process):
    config_schema = {
        'sbml_filepath': 'string',
        'num_points': 'integer',
        'duration': 'integer',
        'start_time': {
            '_type': 'integer',
            '_default': 0
        }
    }

    def __init__(self, sbml_filepath: str, config=None, core=None):
        config = config or {
            'sbml_filepath': sbml_filepath,
        }
        super().__init__(config=config, core=core)

        self.simulator: ExtendedRoadRunner = te.loadSBMLModel(self.config['sbml_filepath'])
        # handle context type (concentrations for deterministic by default)
        context_type = self.config.get('species_context') or 'concentrations'
        self.species_context_key = f'floating_species_{context_type}'
        self.use_counts = 'concentrations' in context_type

        # TODO -- make this configurable.
        self.input_ports = [
            self.species_context_key,
            'boundary_species',
            'model_parameters'
            'time']

        self.output_ports = [
            self.species_context_key,
            'time']

        # Get the species (floating and boundary)
        self.floating_species_list = self.simulator.getFloatingSpeciesIds()
        self.boundary_species_list = self.simulator.getBoundarySpeciesIds()
        self.floating_species_initial = self.simulator.getFloatingSpeciesConcentrations()
        self.boundary_species_initial = self.simulator.getBoundarySpeciesConcentrations()

        # Get the list of parameters and their values
        self.model_parameters_list = self.simulator.getGlobalParameterIds()
        self.model_parameter_values = self.simulator.getGlobalParameterValues()

        # Get a list of reactions
        self.reaction_list = self.simulator.getReactionIds()

    def initial_state(self, config=None):
        floating_species_dict = dict(zip(self.floating_species_list, self.floating_species_initial))
        boundary_species_dict = dict(zip(self.boundary_species_list, self.boundary_species_initial))
        model_parameters_dict = dict(zip(self.model_parameters_list, self.model_parameter_values))
        return {
            'time': 0.0,
            self.species_context_key: floating_species_dict,
            # 'boundary_species': boundary_species_dict,
            'model_parameters': model_parameters_dict
        }

    def inputs(self):
        float_set = {'_type': 'float', '_apply': 'set'}
        return {
            'time': 'float',
            # 'run_time': 'float',
            self.species_context_key: {
                species_id: float_set for species_id in self.floating_species_list},
            # 'boundary_species': {
                # species_id: float_set for species_id in self.boundary_species_list},
            'model_parameters': {
                param_id: float_set for param_id in self.model_parameters_list},
            'reactions': {
                reaction_id: float_set for reaction_id in self.reaction_list},
        }

    def outputs(self):
        float_set = {'_type': 'float', '_apply': 'set'}
        return {
            self.species_context_key: {
                species_id: float_set for species_id in self.floating_species_list},
            'time': 'float'
        }

    def update(self, inputs, interval):

        # set tellurium values according to what is passed in states
        for port_id, values in inputs.items():
            if port_id in self.input_ports:  # only update from input ports
                for cat_id, value in values.items():
                    self.simulator.setValue(cat_id, value)

        for cat_id, value in inputs[self.species_context_key].items():
            self.simulator.setValue(cat_id, value)

        # run the simulation
        result = self.simulator.oneStep(inputs['time'], interval)

        # extract the results and convert to update
        update = {
            'time': interval,  # new_time,
            self.species_context_key: {
                mol_id: float(self.simulator.getValue(mol_id))
                for mol_id in self.floating_species_list
            }
        }

        """for port_id, values in inputs.items():
            if port_id in self.output_ports:
                update[port_id] = {}
                for cat_id in values.keys():
                    update[port_id][cat_id] = self.simulator.getValue(cat_id)"""
        return update
