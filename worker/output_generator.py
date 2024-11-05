from abc import ABC, abstractmethod

import numpy as np
from process_bigraph import Step

from output_data import SBML_EXECUTORS, get_sbml_species_mapping


class OutputGenerator(ABC, Step):
    config_schema = {
        'input_file': 'string',
        'simulator': 'string',
    }

    def __init__(self, config=None, core=None):
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
        'start_time': 'string',
        'end_time': 'string',
        'num_steps': 'integer',
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        if not self.input_file.endswith('.xml'):
            raise ValueError('Input file must be a valid SBML (XML) file')

        self.start_time = self.config.get('start_time')
        self.end_time = self.config.get('end_time')
        self.num_steps = self.config.get('num_steps')
        self.species_mapping = get_sbml_species_mapping(self.input_file)

    def generate(self):
        return SBML_EXECUTORS[self.simulator](self.input_file, self.start_time, self.end_time, self.num_steps)

    def outputs(self):
        return {species_name: 'tree[float]' for species_name in self.species_mapping.keys()}

    def update(self, state):
        return self.generate()
