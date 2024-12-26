from typing import *
from abc import abstractmethod

import numpy as np
from process_bigraph import Step
from biosimulators_processes.processes.copasi_process import CopasiProcess


class ParameterScan(Step):
    """
        For example:

            parameters = {
                'global': {
                    'ADP': <STARTING_VALUE>,
                    ...},
                'species': {
                    'species_a': <STARTING_VALUE>,
                    'species_b': ...
                    ...},
                'reactions': {
                    'R1': {
                        '(R1).k1': <STARTING_VALUE>,
                    ...}
            }

            ...according to the config schema for Copasi Process (model changes)


    """
    config_schema = {
        'n_iterations': 'int',
        'perturbation_magnitude': 'float',
        'parameters': 'tree[string]',  # ie: 'param_type: param_name'... like 'global': 'ADP', 'species': 'S3'
        'process_config': 'tree[string]'}  # use the builder to extract this

    @abstractmethod
    def inputs(self):
        pass

    @abstractmethod
    def outputs(self):
        pass

    @abstractmethod
    def initial_state(self):
        pass

    @abstractmethod
    def update(self, input):
        pass


class DeterministicTimeCourseParameterScan(Step):
    """Using CopasiProcess as the primary TimeCourse simulator.

        # TODO: enable multiple Simulator types.

        should at some point in the stack return a list or dict of configs that can be used by CopasiProcess which are then
            used to run during update
    """
    config_schema = {
        'process_config': 'tree[string]',
        'n_iterations': 'int',
        'iter_stop': 'float',
        'iter_start': 'maybe[float]',
        'perturbation_magnitude': 'float',
        'parameters': 'list[object]'}

    def __init__(self, config=None, core=None):
        self.process = CopasiProcess(config=self.config.get('process_config'))
        self.params_to_scan: List = self.config.get('parameters', [])
        self.n_iterations = self.config['n_iterations']
        self.iter_start = self.config.get('iter_start', 0.0)
        self.iter_stop = self.config['iter_stop']


    def initial_state(self):
        return {
            str(n): self.process.initial_state()
            for n in range(self.n_iterations)}

    def inputs(self):
        return self.process.inputs()

    def outputs(self):
        return {
            str(n): self.process.outputs()
            for n in range(self.n_iterations)}

    def update(self, input):
        """Here is where the method of perturbation differs: deterministic will use
            `np.linspace(...`
        """
        # set up parameters
        results = {}
        scan_range = np.linspace(
            start=self.iter_start,
            stop=self.iter_stop,
            num=self.n_iterations).tolist()

        for index, perturbed in enumerate(scan_range):
            interval = input['time']
            for param in self.params_to_scan:
                if 'global' in param.scope:
                    input_key = 'model_parameters'
                elif 'species' in param.scope:
                    input_key = 'floating_species'
                else:
                    raise ValueError('Only global or species -level parameter scanning is currently supported.')

                for input_id, input_value in input[input_key]:
                    if param.name in input_id:
                        input[input_key][input_id] = perturbed

                r = self.process.update(
                    inputs=input,
                    interval=interval)
                results[str(index)] = r
        return results


class StochasticParameterScan(ParameterScan):
    """Analogous to Monte Carlo perturbations"""
    config_schema = {
        'n_iterations': 'int',
        'perturbation_magnitude': 'float',
        'parameters': 'tree[string]',
        'process_config': 'tree[string]'}

    pass
