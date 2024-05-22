"""
Compare Data Model:
    Objects whose purpose is to compare the output of 'like-minded' simulators/processes
which reside in a shared compositional space. The global 'state' of this composition
is agnostic to any summation of values.


Such engineering should be performed by an expeditionary of semantic unity, using
vocabulary as their protection. The Explorer is truly that: unafraid to step outside of
the unifying 'glossary' in the name of expanding it. Semantics are of both great
use and immense terror to the Explorer. The Explorer firmly understands and believes
these worldly facts.

author: Alex Patrie
license: Apache License, Version 2.0
date: 04/2024
"""


from typing import *
from abc import ABC
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
from pydantic import Field, field_validator

from src import BaseModel, _BaseClass


# TODO: Transpose data frame and make col vectors for each param, where the index is param name,
    # and cols are simulator id.


class ParamIntervalOutputData(BaseModel):
    param_name: str
    value: float


class IntervalOutput(BaseModel):
    interval_id: float
    data: Union[Dict[str, float], List[ParamIntervalOutputData]]


class ParameterScore(BaseModel):
    """Base class for parameter scores in-general."""
    param_name: str
    value: float


class ParameterMSE(ParameterScore):
    """Attribute of Process Parameter RMSE"""
    param_name: str
    value: float = Field(...)  # TODO: Ensure field validation/setting for MSE-specific calculation.
    mean: float
    process_id: str

    @classmethod
    @field_validator('value')
    def set_value(cls, v):
        # TODO: Finish this.
        return v


class ProcessParameterRMSE(BaseModel):
    """Attribute of Process Fitness Score"""
    process_id: str
    param_id: str  # mostly species names or something like that
    value: float  # use calculate rmse here


class ProcessFitnessScore(BaseModel):
    """Attribute of Simulator Process Output Based on the list of interval results"""
    process_id: str
    error: float  # the error by which to bias the rmse calculation
    rmse_values: List[ProcessParameterRMSE]  # indexed by parameter name over whole simulation


class IntervalOutputData(BaseModel):
    """Attribute of Simulator Process Output"""
    param_name: str  # data name
    value: float
    time_id: float  # index for composite run inference
    mse: ParameterMSE


class SimulatorProcessOutput(BaseModel):
    """Attribute of Process Comparison Result"""
    process_id: str
    simulator: str
    data: List[IntervalOutputData]
    fitness_score: ProcessFitnessScore


class ProcessComparisonResult(BaseModel):
    """Generic class inherited for all process comparisons."""
    duration: int
    num_steps: int
    simulators: List[str]
    outputs: List[SimulatorProcessOutput]
    timestamp: str = str(
        datetime.now()) \
        .replace(' ', '_') \
        .replace(':', '-') \
        .replace('.', '-')


# DATA MODEL USED
@dataclass
class ODEProcessIntervalComparison(_BaseClass):
    mse_data: pd.DataFrame
    rmse_data: pd.DataFrame
    inner_prod_data: pd.DataFrame
    outer_prod_data: Dict
    time_id: int 


@dataclass
class ODEIntervalResult(_BaseClass):
    interval_id: float
    copasi_floating_species_concentrations: Dict[str, float]
    tellurium_floating_species_concentrations: Dict[str, float]
    amici_floating_species_concentrations: Dict[str, float]
    time: float


@dataclass
class ODEComparisonResult(_BaseClass):
    duration: int
    num_steps: int
    biomodel_id: str
    timestamp: str
    outputs: Optional[List[ODEIntervalResult]] = None

    def __init__(self, duration, num_steps, biomodel_id):
        super().__init__()
        self.duration = duration
        self.num_steps = num_steps
        self.biomodel_id = biomodel_id
        self.outputs = self._set_outputs()
        self.timestamp = self._set_timestamp()

    @classmethod
    def _set_timestamp(cls):
        return str(datetime.now()).replace(' ', '_').replace(':', '-').replace('.', '-')

    def _set_outputs(self):
        return self.generate_ode_interval_outputs(
            self.duration,
            self.num_steps,
            self.biomodel_id)

    def generate_ode_interval_outputs(self, duration: int, n_steps: int, biomodel_id: str) -> List[ODEIntervalResult]:
        def _generate_ode_interval_results(duration: int, n_steps: int, biomodel_id: str) -> List[ODEIntervalResult]:
            results_dict = self.generate_ode_comparison(biomodel_id, duration)
            simulator_names = ['copasi', 'tellurium', 'amici']
            interval_results = []

            for global_time_index, interval_result_data in enumerate(results_dict['outputs']):
                interval_config = {
                    'interval_id': float(global_time_index),
                    'time': interval_result_data['time']
                }

                for k, v in interval_result_data.items():
                    for simulator_name in simulator_names:
                        if simulator_name in k:
                            interval_config[f'{simulator_name}_floating_species_concentrations'] = v

                interval_result = ODEIntervalResult(**interval_config)
                interval_results.append(interval_result)

            return interval_results

        return _generate_ode_interval_results(duration, n_steps, biomodel_id)

    @classmethod
    def generate_ode_comparison(cls, biomodel_id: str, dur: int) -> Dict:
        """Run the `compare_ode_step` composite and return data which encapsulates another composite
            workflow specified by dir.

            Args:
                biomodel_id:`str`: A Valid Biomodel ID.
                dur:`int`: duration of the internal composite simulation.

            Returns:
                `Dict` of simulation comparison results like `{'outputs': {...etc}}`
        """
        compare = {
            'compare_ode': {
                '_type': 'step',
                'address': 'local:compare_ode_step',
                'config': {'biomodel_id': biomodel_id, 'duration': dur},
                'inputs': {},
                'outputs': {'comparison_data': ['comparison_store']}
            },
            'verification_data': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'emit': {'comparison_data': 'tree[any]'}
                },
                'inputs': {'comparison_data': ['comparison_store']}
            }
        }

        wf = Composite(config={'state': compare}, core=CORE)
        wf.run(1)
        comparison_results = wf.gather_results()
        output = comparison_results[("verification_data"),][0]['comparison_data']

        return {'outputs': output[('emitter',)]}


@dataclass
class ODECompositionResult(_BaseClass):
    """Generalized class for composition results.  TODO: switch to using this instead of ode comparison result. """
    duration: int
    num_steps: int
    model_entrypoint: str  # One of: biomodel id or sbml fp
    simulator_names: List[str]
    timestamp: Optional[str] = str(datetime.now()).replace(' ', '-').replace(':', '_').replace('.', '-')
    outputs: Optional[List[ODEIntervalResult]] = None

    def __init__(self, duration, num_steps, model_entrypoint, simulator_names):
        super().__init__()
        self.duration = duration
        self.num_steps = num_steps
        self.model_entrypoint = model_entrypoint
        self.simulator_names = simulator_names
        self.outputs = self._set_outputs()

    def _set_outputs(self):
        return self.generate_ode_interval_outputs(
            self.duration,
            self.num_steps)

    def generate_ode_interval_outputs(self, duration: int, n_steps: int) -> List[ODEIntervalResult]:
        return self._generate_ode_interval_results(duration, n_steps)

    def _generate_ode_interval_results(self, duration: int, n_steps: int) -> List[ODEIntervalResult]:
        results_dict = self.generate_ode_comparison(self.model_entrypoint, duration)
        interval_results = []

        for global_time_index, interval_result_data in enumerate(results_dict['outputs']):
            interval_config = {
                'interval_id': float(global_time_index),
                'time': interval_result_data['time']
            }

            for k, v in interval_result_data.items():
                for simulator_name in self.simulator_names:
                    if simulator_name in k:
                        interval_config[f'{simulator_name}_floating_species_concentrations'] = v

            interval_result = ODEIntervalResult(**interval_config)
            interval_results.append(interval_result)

        return interval_results

    @classmethod
    def generate_ode_comparison(cls, model_entrypoint: str, dur: int) -> Dict:
        """Run the `compare_ode_step` composite and return data which encapsulates another composite
            workflow specified by dir.

            Args:
                model_entrypoint:`str`: A Valid Biomodel ID.
                dur:`int`: duration of the internal composite simulation.

            Returns:
                `Dict` of simulation comparison results like `{'outputs': {...etc}}`
        """
        compare = {
            'compare_ode': {
                '_type': 'step',
                'address': 'local:compare_ode_step',
                'config': {'model_entrypoint': model_entrypoint, 'duration': dur},
                'inputs': {},
                'outputs': {'comparison_data': ['comparison_store']}
            },
            'verification_data': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'emit': {'comparison_data': 'tree[any]'}
                },
                'inputs': {'comparison_data': ['comparison_store']}
            }
        }

        wf = Composite(config={'state': compare}, core=CORE)
        wf.run(1)
        comparison_results = wf.gather_results()
        output = comparison_results[("verification_data"),][0]['comparison_data']

        return {'outputs': output[('emitter',)]}


class CompositeRunError(BaseModel):
    exception: Exception


class ComparisonDocument(ABC):
    def __init__(self):
        pass


class ODEComparisonDocument(ComparisonDocument):
    """To be called 'behind-the-scenes' by the Comparison REST API"""
    def __init__(self,
                 duration: int,
                 num_steps: int,
                 model_filepath: str,
                 framework_type='deterministic',
                 simulators: Optional[Union[List[str], Dict[str, str]]] = None,
                 target_parameter: Optional[Dict[str, Union[str, float]]] = None,
                 **kwargs):
        """This object implements a self generated factory with which it creates its representation. The naming of
            simulator processes within the composition are by default generated through concatenating the simulator
            tool _name_(i.e: `'tellurium'`) with with a simple index `i` which is a population of an iteration over
            the total number of processes in the bigraph.

                Args:
                    simulators:`Union[List[str], Dict[str, str]]`: either a list of actual simulator tool names,
                        ie: `'copasi'`; or a dict mapping of {simulator_tool_name: custom_process_id}
                    duration:`int`: the total duration of simulation run
                    num_steps:`int`
                    model_filepath:`str`: filepath which points to a SBML model file.
                    framework_type:`str`: type of mathematical framework to employ with the simulators within your
                        composition. Choices are `'stochastic'`, `'deterministic'`. Note that there may be more
                        stochastic options than deterministic.
        """
        super().__init__()

        if simulators is None:
            self.simulators = ['tellurium', 'copasi', 'amici']
        elif isinstance(simulators, dict):
            self.simulators = list(simulators.keys()) if isinstance(simulators, dict) else simulators
            self.custom_process_ids = list(simulators.values())
        else:
            self.simulators = simulators

        self.composite = kwargs.get('composite', {})
        self.framework_type = framework_type

        context = 'concentrations'
        self.species_port_name = f'floating_species_{context}'
        self.species_store = [f'floating_species_{context}_store']
        self._populate_composition(model_filepath)

    def add_single_process_to_composite(self, process_id: str, simulator: str):
        process_instance = prepare_single_ode_process_document(
            process_id=process_id,
            simulator_name=simulator,
            sbml_model_fp=self.model_filepath,
            add_emitter=False)
        self.composite[process_id] = process_instance[process_id]

    def _generate_composite_index(self) -> float:
        # TODO: implement this.
        pass

    def _add_emitter(self) -> None:  # TODO: How do we reference different nesting levels?
        self.composite['emitter'] = {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    self.species_port_name: 'tree[float]',
                    'time': 'float'}
            },
            'inputs': {
                self.species_port_name: self.species_store,
                'time': ['time_store']}}

    def _populate_composition(self, model_filepath: str):
        context = 'concentrations'
        for index, process in enumerate(self.simulators):
            self._add_ode_process_schema(
                process_name=process,
                species_context=context,
                i=index,
                model={'model_source': model_filepath})
        return self._add_emitter()

    def _add_ode_process_schema(
            self,
            process_name: str,
            species_context: str,
            i: int,
            **config
    ) -> None:
        species_port_name = f'floating_species_{species_context}'
        species_store = [f'floating_species_{species_context}_store']
        self.composite[f'{process_name}_{i}'] = {
            '_type': 'process',
            'address': f'local:{process_name}',
            'config': config,
            'inputs': {
                species_port_name: species_store,
                'model_parameters': ['model_parameters_store'],
                'time': ['time_store'],
                'reactions': ['reactions_store']
            },
            'outputs': {
                species_port_name: species_store,
                'time': ['time_store']
            }
        }


class DocumentFactory:
    @classmethod
    def from_dict(cls, configuration: Dict) -> ComparisonDocument:
        """
            Args:
                configuration:`Dict`: required keys:
                     simulators: List[str],
                     duration: int,
                     num_steps: int,
                     model_filepath: str,
                     framework_type='deterministic',
                     target_parameter: Dict[str, Union[str, float]] = None

        """
        return ComparisonDocument(**configuration)
