import inspect
import copy
import json
import types
import abc
from abc import ABC
from functools import reduce
import operator

import numpy as np
from bigraph_viz import plot_bigraph, pf, pp
from bigraph_viz.dict_utils import schema_keys

from sed2.dict_utils import deep_merge

schema_keys.extend(['_type', 'config'])

"""
Decorators
==========
"""


def register(identifier, registry):
    def decorator(func):
        registry.register(func, identifier=identifier)
        return func

    return decorator


def annotate(annotation):
    def decorator(func):
        func.annotation = annotation
        return func

    return decorator


# TODO: ports for functions require input/output, but for processes this isn't required
# TODO assert type are in type_registry
# TODO check that keys match function signature
def ports(ports_schema):
    # assert inputs/outputs and types, give suggestions
    allowed = ['inputs', 'outputs']
    assert all(key in allowed for key in
               ports_schema.keys()), f'{[key for key in ports_schema.keys() if key not in allowed]} not allowed as top-level port keys. Allowed keys include {str(allowed)}'
    process_ports = copy.deepcopy(ports_schema.get('inputs', {}))
    process_ports.update(ports_schema.get('outputs', {}))

    def decorator(func):
        func.input_output_ports = ports_schema
        func.ports = process_ports
        return func

    return decorator


"""
Registry
========
"""


class ProcessRegistry:
    def __init__(self):
        self.registry = {}

    def register(self, process, identifier=None):
        if not identifier:
            identifier = process.__name__
        # signature = inspect.signature(process)
        annotation = getattr(process, 'annotation', None)
        process_ports = getattr(process, 'ports')

        try:
            bases = [base.__name__ for base in process.__bases__]
        except:
            bases = None

        process_class = None
        if isinstance(process, types.FunctionType):
            process_class = 'function'
        elif 'Composite' in bases:
            process_class = 'composite'
        elif 'Process' in bases:
            process_class = 'process'
        process.process_class = process_class  # add process class annotation

        # TODO -- assert ports and signature match
        if not process_ports:
            raise Exception(f'Process {identifier} requires ports')

        item = {
            'annotation': annotation,
            'ports': process_ports,
            'address': process,
            'class': process_class,
        }
        self.registry[identifier] = item

    def access(self, name):
        if name not in self.registry:
            raise Exception(f'{name} not in the registry')
        return self.registry.get(name)

    def get_annotations(self):
        return [v.get('annotation') for k, v in self.registry.items()]

    def activate_process(self, process_name, namespace):
        namespace[process_name] = self.registry[process_name]['address']

    def activate_all(self, namespace):
        """how to add to globals: process_registry.activate_all(globals())"""
        for process_name in self.registry.keys():
            self.activate_process(process_name, namespace)


# initialize a registry
sed_process_registry = ProcessRegistry()

"""
More helper functions
=====================
"""


def serialize_instance(wiring):
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f'Object of type "{obj.__class__.__name__}" is not JSON serializable')

    return json.dumps(wiring, default=convert_numpy)


def deserialize_instance(serialized_wiring):
    if isinstance(serialized_wiring, dict):
        return serialized_wiring

    def convert_numpy(obj):
        if isinstance(obj, list):
            return np.array(obj)
        return obj

    return json.loads(serialized_wiring, object_hook=convert_numpy)


def get_value_from_path(dictionary, path):
    # noinspection PyBroadException
    try:
        return reduce(operator.getitem, path, dictionary)
    except Exception:
        return None


def extract_composite_config(schema):
    config = {k: v for k, v in schema.items() if k not in schema_keys}
    return config


def get_processes_states_from_schema(schema, process_registry, path=None):
    schema = copy.deepcopy(schema)
    path = path or ()

    processes = {}
    states = {}
    for name, value in schema.items():
        if name in schema_keys:
            continue

        next_path = path + (name,)
        if isinstance(value, dict) and value.get('wires'):
            # get the process
            process_class = value.pop('_type')
            process_wires = value.pop('wires')
            process_depends_on = value.get('_depends_on', [])
            process = process_registry.access(process_class)
            try:
                process_ports = process['ports']
            except:
                raise Exception(f'process {process} has no ports')

            assert process_wires.keys() == process_ports.keys(), f'{name} wires {list(process_wires.keys())} ' \
                                                                 f'need to match ports {list(process_ports.keys())}'
            # initialize the process
            process_class = process['class']
            process_address = process['address']
            if process_class == 'function':
                actual_process = process_address
            elif process_class == 'process':
                actual_process = process_address(value)
            elif process_class == 'composite':
                actual_process = process_address(value,
                                                 process_registry)  # TODO -- get process config, not full value
                value = {}

            # set depends on
            # actual_process.depends_on = process_depends_on
            processes[next_path] = {'address': actual_process, '_depends_on': process_depends_on}

            # initialize connected states
            for port_id, wire in process_wires.items():
                if wire not in states:
                    port_type = process_ports[port_id]
                    states.update({wire: None})

            p, s = get_processes_states_from_schema(value, process_registry, path=next_path)
            processes.update(p)
            for k, v in s.items():
                if states.get(k) is None:  # not None
                    states.update(s)

        else:
            states[name] = value

    return processes, states


"""
Process and Composite base classes
==================================
"""


def topological_sort(graph):
    """Return list of sorted process names based on dependencies"""
    visited = set()
    sorted_list = []

    def visit(path):
        if path not in visited:
            visited.add(path)
            if path in graph:
                depends_on = graph[path]['_depends_on']
                if depends_on:
                    for neighbor in depends_on:
                        if not isinstance(neighbor, tuple):
                            neighbor = (neighbor,)
                        visit(neighbor)
            sorted_list.append(path)

    for path, node in graph.items():
        visit(path)
    return sorted_list


class Process:
    config = {}
    ports = None

    def __init__(self, config):
        self.initialize_process(config)

    def initialize_process(self, config):
        self.config = config
        if self.config.get('_ports'):
            self.ports = self.config.get('_ports')

    def get_ports(self):
        assert isinstance(self.ports, dict)
        return self.ports

    @abc.abstractmethod
    def update(self, state):
        return {}


class Composite(Process, ABC):
    config = {}
    processes = None
    states = None
    process_registry = None

    def __init__(self, config, process_registry):
        self.initialize_process(config)
        self.initialize_composite(process_registry)

    def initialize_composite(self, process_registry):
        self.process_registry = process_registry
        processes, states = get_processes_states_from_schema(
            self.config, self.process_registry)
        self.states = states
        self.processes = processes
        self.sorted_processes = topological_sort(processes)

    def get_process_wires(self, process_path):
        process_value = get_value_from_path(self.config, process_path)
        return process_value['wires']

    def process_state(self, process_path):
        # get the states for this specific process
        wires = self.get_process_wires(process_path)
        return {wire_id: self.states[target] for wire_id, target in wires.items()}

    def inverse_topology(self, process_path, state):
        wires = self.get_process_wires(process_path)
        return {wires[port]: v for port, v in state.items()}

    def to_json(self):
        return serialize_instance(self.config)

    def apply(self, result):
        for k, v in result.items():
            self.states[k] = v  # this is strictly "set" apply method. TODO -- use an apply_registry

    def update_process(self, process_path, state):
        process = self.processes[process_path]['address']
        process_states = self.process_state(process_path)

        # process_states = deep_merge(process_states, state)  # TODO -- update process_states with state. need to inverse topology
        if process.process_class == 'function':
            input_states = {
                k: process_states[k] for k in process.input_output_ports['inputs'].keys()}
            raw_update = process(**input_states)
            if not isinstance(raw_update, list) and not isinstance(raw_update, tuple):
                raw_update = [raw_update]
            update = {
                k: raw_update[idx]
                for idx, k in enumerate(process.input_output_ports.get('outputs', {}).keys())}
        else:
            update = process.update(state=process_states)

        absolute_update = self.inverse_topology(process_path, update)
        self.apply(absolute_update)

    def update(self, state=None):
        for process_path in self.sorted_processes:
            self.update_process(process_path, state)
        return {
            port: self.states[port]
            for port in self.config.get('wires', {}).keys()
            if port in self.states}


"""
Register example processes and composites
=========================================
"""


@register(
    identifier='control:range_iterator',
    registry=sed_process_registry)
@ports({
    'inputs': {
        'trials': 'int',
    },
    'outputs': {
        'results': 'list'}})
@annotate('more info here?')
class RangeIterator(Composite):
    def update(self, state):
        trials = state.get('trials', 0)
        for i in range(trials):
            for process_path, process in self.processes.items():
                self.update_process(process_path, state)
        return {
            'results': self.states['value'],
            'trials': 0
        }


@register(
    identifier='control:range_iterator:model',
    registry=sed_process_registry)
@ports({
    'inputs': {
        'trials': 'int',
        'model_instance': 'Model',
    },
    'outputs': {
        'results': 'list'}})
@annotate('more info here?')
class RangeIteratorModel(Composite):
    def update(self, state):
        trials = state.get('trials', 0)
        self.states['model_instance'] = state['model_instance']  # move model in
        for i in range(trials):
            for process_path, process in self.processes.items():
                self.update_process(process_path, state)
        return {
            'results': self.states['results'],
            # 'trials': 0
        }


@register(
    identifier='math:sum_list',
    registry=sed_process_registry)
@ports({
    'inputs': {'values': 'list[float]'},
    'outputs': {'result': 'float'}})
@annotate('more info here?')
def add_list(values):
    if not isinstance(values, list):
        values = [values]
    return sum(values)


@register(
    identifier='math:add_two',
    registry=sed_process_registry)
@ports({
    'inputs': {'a': 'float', 'b': 'float'},
    'outputs': {'result': 'float'}})
@annotate('more info here?')
def add_two(a, b):
    return a + b


"""
Examples
========
"""


def run_instance1():
    config1 = {
        # top-level state
        'trials': 10,
        'results': None,  # this should be filled in automatically

        # a composite process
        'for_loop': {
            '_type': 'control:range_iterator',
            'wires': {
                'trials': 'trials',
                'results': 'results',
            },

            # state within for_loop
            'value': 0,
            'added': 1,

            # process within for_loop
            'add': {
                '_type': 'math:add_two',
                'wires': {
                    'a': 'value',
                    'b': 'added',
                    'result': 'value',
                },
            }
        },
        'wires': {
            'results': 'results',
            'trials': 'trials',
        }
    }

    sim_experiment = Composite(
        config=config1,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)
    print(results)

    plot_bigraph(config1, out_dir='../composites/out', filename='test1')


if __name__ == '__main__':
    run_instance1()
