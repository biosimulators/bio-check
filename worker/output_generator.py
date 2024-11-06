from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import numpy as np
from attr import dataclass
from process_bigraph import Step, ProcessTypes

from output_data import SBML_EXECUTORS, get_sbml_species_mapping


CORE = ProcessTypes()


class OutputGenerator(Step):
    config_schema = {
        'input_file': 'string',
        'context': 'string',
    }

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        self.input_file = self.config['input_file']
        self.context = self.config.get('context')
        if self.context is None:
            raise ValueError("context(simulator) must be specified in this processes' config")

    @abstractmethod
    def generate(self, parameters: Optional[Dict[str, Any]] = None):
        """Abstract method for generating output data upon which to base analysis from based on its origin.

        This can be used for logic of any scope.
        NOTE: args and kwargs are not defined in this function, but rather should be defined by the
        inheriting class' constructor: i,e; start_time, etc.

        Kwargs relate only to the given simulator api you are working with.
        """
        pass

    def initial_state(self):
        # base class method
        return {
            'output_data': {}
        }

    def inputs(self):
        return {
            'parameters': 'tree[any]'
        }

    def outputs(self):
        return {
            'output_data': 'tree[any]'
        }

    def update(self, state):
        parameters = state.get('parameters') if isinstance(state, dict) else {}
        data = self.generate(parameters)
        return {'output_data': data}


CORE.process_registry.register('output-generator', OutputGenerator)


class TimeCourseOutputGenerator(OutputGenerator):
    # NOTE: we include defaults here as opposed to constructor for the purpose of deliberate declaration within .json state representation.
    config_schema = {
        # 'input_file': 'string',
        # 'context': 'string',
        'start_time': {
            '_type': 'integer',
            '_default': 0
        },
        'end_time': {
            '_type': 'integer',
            '_default': 10
        },
        'num_steps': {
            '_type': 'integer',
            '_default': 100
        },
    }

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        if not self.input_file.endswith('.xml'):
            raise ValueError('Input file must be a valid SBML (XML) file')

        self.start_time = self.config.get('start_time')
        self.end_time = self.config.get('end_time')
        self.num_steps = self.config.get('num_steps')
        self.species_mapping = get_sbml_species_mapping(self.input_file)

    def initial_state(self):
        # TODO: implement this
        pass

    def generate(self, parameters: Optional[Dict[str, Any]] = None):
        # TODO: add kwargs (initial state specs) here
        executor = SBML_EXECUTORS[self.context]
        data = executor(self.input_file, self.start_time, self.end_time, self.num_steps)
        return data


CORE.process_registry.register('time-course-output-generator', TimeCourseOutputGenerator)


@dataclass
class NodeSpec:
    name: str
    _type: str
    address: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]


@dataclass
class StepNodeSpec(NodeSpec):
    _type: str = "step"


@dataclass
class ProcessNodeSpec(NodeSpec):
    _type: str = "process"


def node_spec(name: str, _type: str, address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any]):
    return {
        name: {
            '_type': _type,
            'address': address,
            'config': config,
            'inputs': inputs,
            'outputs': outputs
        }
    }


def step_node_spec(name: str, address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any]):
    return node_spec(name=name, _type="step", address=address, config=config, inputs=inputs, outputs=outputs)


def process_node_spec(name: str, address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any]):
    return node_spec(name=name, _type="process", address=address, config=config, inputs=inputs, outputs=outputs)


def time_course_node_spec(name: str, input_file: str, context: str, start_time: int, end_time: int, num_steps: int):
    config = {
        'input_file': input_file,
        'start_time': start_time,
        'end_time': end_time,
        'num_steps': num_steps,
        'context': context
    }
    return step_node_spec(
        name=name,
        address='time-course-output-generator',
        config=config,
        inputs={'parameters': [f'parameters_store_{name}']},
        outputs={'output_data': [f'output_data_store_{name}']}
    )


def generate_time_course_data(input_fp: str, start: int, dur: int, steps: int, simulators: List[str] = None, expected_results_fp: str = None):
    pass

# def generate_time_course_data():
#     doc = {
#         'copasi': {
#             '_type': 'step',
#             'address': 'local:time-course-output-generator',
#             'config': {
#                 'input_file': TEST_SBML_FP,
#                 'context': 'copasi',
#                 'start_time': 0,
#                 'end_time': 10,
#                 'num_steps': 100
#             },
#             'inputs': {
#                 'parameters': ['parameters_store']
#             },
#             'outputs': {
#                 'output_data': ['output_data_store'],
#             }
#         }
#     }
#     sim = Composite({
#             'state': doc,
#             'emitter': {'mode': 'all'}
#         },
#         core=CORE
#     )
#     sim.save(filename="test_time_course_output_generator_before.json", outdir="./outputs")
#     print(dir(sim))
#     sim.run(1)
#     results = sim.gather_results()
#     print(f'Results:\n{results}')
#     sim.save(filename="test_utc_output_generator_after.json", outdir="./outputs")
#     return sim