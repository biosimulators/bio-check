"""
Tellurium Process
"""


import numpy as np
import tellurium as te
from process_bigraph import Process, Composite, pf, Step

from biosimulators_processes import CORE
from biosimulators_processes.processes.utc_process import SbmlUniformTimeCourse
from biosimulators_processes.data_model.sed_data_model import MODEL_TYPE


class UtcTellurium(SbmlUniformTimeCourse):
    def __init__(self,
                 config=None,
                 core=CORE,
                 time_config: dict = None,
                 model_source: str = None,
                 sed_model_config: dict = None):
        super().__init__(config, core, time_config, model_source, sed_model_config)
        self.floating_species_list = self.simulator.getFloatingSpeciesIds()
        self.boundary_species_list = self.simulator.getBoundarySpeciesIds()
        self.floating_species_initial = self.simulator.getFloatingSpeciesConcentrations()
        self.boundary_species_initial = self.simulator.getBoundarySpeciesConcentrations()

        # Get the list of parameters and their values
        self.model_parameters_list = self.simulator.getGlobalParameterIds()
        self.model_parameter_values = self.simulator.getGlobalParameterValues()

        # Get a list of reactions
        self.reaction_list = self.simulator.getReactionIds()

    def _get_initial_state_params(self):
        floating_species_dict = dict(zip(self.floating_species_list, self.floating_species_initial))
        # boundary_species_dict = dict(zip(self.boundary_species_list, self.boundary_species_initial))
        model_parameters_dict = dict(zip(self.model_parameters_list, self.model_parameter_values))
        return {
            'time': [0.0],
            self.species_context_key: floating_species_dict,
            'model_parameters': model_parameters_dict,
            'reactions': self.reaction_list
        }

    def plot_results(self):
        # return self.simulator.plot()
        return super().plot_results(simulator_name='Tellurium')

    def _load_simulator(self, model_fp: str, **kwargs):
        return te.loadSBMLModel(model_fp)

    def _get_reactions(self) -> list[str]:
        return self.simulator.getReactionIds()

    def _get_floating_species(self) -> list[str]:
        return self.simulator.getFloatingSpeciesIds()

    def _get_model_parameters(self) -> list[str]:
        return self.simulator.getGlobalParameterIds()

    def _generate_results(self, inputs=None):
        x = inputs or self.initial_state()
        # TODO: set vals if inputs here.
        output_start = self.output_start_time + 1 if self.output_start_time < 1 else self.output_start_time
        self.simulator.simulate(start=self.initial_time, end=output_start)
        s = self.simulator.simulate(start=output_start, end=self.duration, steps=self.num_steps)
        outputs = {'floating_species': {}}
        for i, row in enumerate(s.transpose()):
            if i < 1:
                outputs['time'] = row
            else:
                for spec_index, name in enumerate(self.floating_species_list):
                    outputs['floating_species'][self.output_keys[spec_index]] = row
        return outputs


class TelluriumStep(Step):

    config_schema = {
        'sbml_model_path': 'string',
        'antimony_string': 'string',
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        # initialize a tellurium(roadrunner) simulation object. Load the model in using either sbml(default) or antimony
        if self.config.get('antimony_string') and not self.config.get('sbml_model_path'):
            self.simulator = te.loada(self.config['antimony_string'])
        elif self.config.get('sbml_model_path') and not self.config.get('antimony_string'):
            self.simulator: te.roadrunner.extended_roadrunner.ExtendedRoadRunner = te.loadSBMLModel(self.config['sbml_model_path'])
        else:
            raise Exception('the config requires either an "antimony_string" or an "sbml_model_path"')

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

    # TODO -- is initial state even working for steps?
    def initial_state(self, config=None):
        return {
            'inputs': {
                'time': 0,
            },
        }

    def inputs(self):
        return {
            'time': 'float',
            'run_time': 'float',
        }

    def outputs(self):
        return {
            'results': {
                '_type': 'numpy_array',
                '_apply': 'set'
            }  # This is a roadrunner._roadrunner.NamedArray
        }

    def update(self, inputs):
        results = self.simulator.simulate(
            inputs['time'],
            inputs['run_time'],
            10
        )  # TODO -- adjust the number of saves teps
        return {
            'results': results}


class TelluriumProcess(Process):
    config_schema = {
        'record_history': 'maybe[boolean]',  # TODO -- do we have this type?
        'model': MODEL_TYPE,
        'method': {
            '_default': 'cvode',
            '_type': 'string'
        },
        'species_context': {
            '_default': 'concentrations',
            '_type': 'string'
        },
        'num_steps': 'maybe[integer]'
    }

    simulator: te.roadrunner.extended_roadrunner.ExtendedRoadRunner
    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        model_source = self.config['model']['model_source']
        if '/' in model_source:
            self.simulator = te.loadSBMLModel(model_source)
        else:
            if 'model' in model_source:  # TODO: find a better way to do this
                self.simulator = te.loada(model_source)
            else:
                raise Exception('the config requires either an "antimony_string" or an "sbml_model_path"')

        # handle context type (concentrations for deterministic by default)
        context_type = self.config['species_context']
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


def test_process():
    # 1. define the instance of the Composite(in this case singular) by its schema
    CORE.process_registry.register('biosimulators_processes.processes.tellurium_process.TelluriumProcess', TelluriumProcess)
    instance = {
        'tellurium': {
            '_type': 'process',
            'address': 'local:!biosimulators_processes.processes.tellurium_process.TelluriumProcess', # using a local toy process
            'config': {
                'sbml_model_path': 'biosimulators_processes/model_files/BIOMD0000000061_url.xml',
            },
            'inputs': {
                'time': ['start_time_store'],
                'run_time': ['run_time_store'],
                'floating_species': ['floating_species_store'],
                'boundary_species': ['boundary_species_store'],
                'model_parameters': ['model_parameters_store'],
                'reactions': ['reactions_store'],
                # 'interval': ['interval_store'],
            },
            'outputs': {
                'results': ['results_store'],
            }
        }
    }

    # 2. make the composite
    workflow = Composite({
        'state': instance
    })

    # 3. run
    # update = workflow.run(10)

    # 4. gather results
    # results = workflow.gather_results()
    # print(f'RESULTS: {pf(results)}')
