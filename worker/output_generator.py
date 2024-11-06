import json
import os
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

import numpy as np
from attr import dataclass
from process_bigraph import Step, ProcessTypes, Composite

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


class NodeSpec(dict):
    _type: str
    address: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    name: Optional[str] = None

    def __init__(self, _type, address, config, inputs, outputs, name=None):
        self._type = _type
        self.address = address
        self.config = config
        self.inputs = inputs
        self.outputs = outputs
        self.name = name


class StepNodeSpec(NodeSpec):
    def __init__(self, address, config, inputs, outputs, name=None):
        super().__init__("step", address, config, inputs, outputs)


@dataclass
class ProcessNodeSpec(NodeSpec):
    _type: str = "process"


def node_spec(_type: str, address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any], name: str = None):
    spec = {
        '_type': _type,
        'address': address,
        'config': config,
        'inputs': inputs,
        'outputs': outputs
    }

    return {name: spec} if name else spec


def step_node_spec(address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any], name: str = None):
    return node_spec(name=name, _type="step", address=address, config=config, inputs=inputs, outputs=outputs)


def process_node_spec(address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any], name: str = None):
    return node_spec(name=name, _type="process", address=address, config=config, inputs=inputs, outputs=outputs)


def time_course_node_spec(input_file: str, context: str, start_time: int, end_time: int, num_steps: int):
    config = {
        'input_file': input_file,
        'start_time': start_time,
        'end_time': end_time,
        'num_steps': num_steps,
        'context': context
    }
    return step_node_spec(
        address='local:time-course-output-generator',
        config=config,
        inputs={
            'parameters': [f'parameters_store_{context}']
        },
        outputs={
            'output_data': [f'output_data_store_{context}']
        }
    )


def generate_time_course_data(
        input_fp: str,
        start: int,
        end: int,
        steps: int,
        simulators: List[str] = None,
        parameters: Dict[str, Any] = None,
        expected_results_fp: str = None,
        out_dir: str = None):
    requested_sims = simulators or ["amici", "copasi", "pysces", "tellurium"]
    spec = {
        simulator: time_course_node_spec(
            input_file=input_fp,
            context=simulator,
            start_time=start,
            end_time=end,
            num_steps=steps
        ) for simulator in requested_sims
    }
    with open('time-course-spec.json', 'w') as f:
        json.dump(spec, f)
    simulation = Composite({'state': spec, 'emitter': {'mode': 'all'}}, core=CORE)
    if out_dir:
        simulation.save(
            filename='time-course-initialization.json',
            outdir=out_dir
        )

    # TODO: is there a better way to do this? (interval of one? Is that symbolic more than anything?)
    if parameters:
        simulation.update(parameters, 1)
    else:
        simulation.run(1)
    if out_dir:
        simulation.save(
            filename='time-course-update.json',
            outdir=out_dir
        )

    output_data = {}
    raw_data = simulation.gather_results()[('emitter',)]
    for data in raw_data:
        for data_key, data_value in data.items():
            if data_key.startswith('output_data_store_'):
                simulator = data_key.split('_')[-1]
                output_data[simulator] = data_value

    return output_data


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