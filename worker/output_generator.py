from abc import ABC, abstractmethod

import numpy as np
from process_bigraph import Step, ProcessTypes

from output_data import SBML_EXECUTORS, get_sbml_species_mapping


CORE = ProcessTypes()


class OutputGenerator(ABC, Step):
    config_schema = {
        'input_file': 'string',
        'simulator': 'string',
    }

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        self.input_file = self.config['input_file']
        self.simulator = self.config.get('simulator')
        if self.simulator is None:
            raise ValueError('Simulator must be specified')

    @abstractmethod
    def generate(self):
        """Abstract method for generating output data upon which to base analysis from based on its origin.

        NOTE: args and kwargs are not defined in this function, but rather should be defined by the
        inheriting class' constructor: i,e; start_time, etc.
        """
        pass


class UtcOutputGenerator(OutputGenerator):
    config_schema = {
        'input_file': 'string',
        'simulator': 'string',
        'start_time': 'integer',
        'end_time': 'integer',
        'num_steps': 'integer',
    }

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        if not self.input_file.endswith('.xml'):
            raise ValueError('Input file must be a valid SBML (XML) file')

        self.start_time = self.config.get('start_time', 0)
        self.end_time = self.config.get('end_time', 10)
        self.num_steps = self.config.get('num_steps', 100)
        self.species_mapping = get_sbml_species_mapping(self.input_file)

    def generate(self, state=None):
        state = state or {}
        input_file = state.get('input_file', self.input_file)
        start = state.get('start_time', self.start_time)
        end = state.get('end_time', self.end_time)
        num_steps = state.get('num_steps', self.num_steps)

        return SBML_EXECUTORS[self.simulator](input_file, start, end, num_steps)

    def inputs(self):
        return {
            'input_file': 'maybe[string]',
            'start_time': 'maybe[integer]',
            'end_time': 'maybe[integer]',
            'num_steps': 'maybe[integer]',
        }

    def outputs(self):
        return {
            'time_course_data': 'tree[float]'
        }

    def update(self, state):
        return self.generate(state)


CORE.process_registry.register('output_generator', OutputGenerator)
CORE.process_registry.register('utc_output_generator', UtcOutputGenerator)
