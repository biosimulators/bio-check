"""
Functions for comparing two or more data sets.

author: Alex Patrie
license: Apache License, Version 2.0
date: 04/2024
"""

from tempfile import mkdtemp
from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import *

import numpy as np
import pandas as pd
from process_bigraph import Step, pp, Composite

from src import CORE
from src.data_model.comparisons import SimulatorComparison, ComparisonMatrix


# process-bigraph classes
class SimulatorComparisonStep(ABC, Step):
    config_schema = {
        'model_entrypoint': 'string',  # either biomodel id or sbml filepath TODO: make from string.
        'duration': 'integer',
        'simulators': {
            '_type': 'list[string]',
            '_default': []}}

    model_entrypoint: str
    duration: int

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        source = config['model_entrypoint']
        assert 'BIO' in source or '/' in source, "You must enter either a biomodel id or path to an sbml model."

    def inputs(self):
        return {}

    def outputs(self):
        return {'comparison_data': 'tree[any]'}

    def update(self, state):
        results = self._run_composition()
        output = {'comparison_data': results}
        return output

    @abstractmethod
    def _run_composition(self, comp: Composite) -> Dict:
        pass


class OdeComparisonStep(SimulatorComparisonStep):
    """TODO: Use the spec's entrypoint arguments as config attributes here!!!

        1. take in a dur, model/archive, simulation, comparison_method, ground_truth, and more (optional simulators)
        2. with the specified entrypoint, iterate over simulators and
            load model instances from tellurium. copasi, and amici, respectively.
        3. in constructor, create mapping of loaded simulators to their names.
        4. Create 3 seperate methods for updating/setting: tellurium_update, copasi_update, amici_update
        5. Call the three simulator methods in parallel for a "dur" on update.
        6. Return just the df from the comparison step via the output ports.
        7. Closer to the API, make a function that calls this step and uses its outputs as one of the
            parameters to instantiate and return the `ComparisonMatrix`.
    """

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        self.model_source = self.config['model_entrypoint']
        self.duration = self.config['duration']

        model_fp = self.model_source if not self.model_source.startswith('BIO') else fetch_biomodel_sbml_file(self.model_source, save_dir=mkdtemp())
        self.document = {
            'copasi_simple': {
                '_type': 'process',
                'address': 'local:copasi',
                'config': {'model': {'model_source': model_fp}},
                'inputs': {'floating_species_concentrations': ['copasi_simple_floating_species_concentrations_store'],
                           'model_parameters': ['model_parameters_store'],
                           'time': ['time_store'],
                           'reactions': ['reactions_store']},
                'outputs': {'floating_species_concentrations': ['copasi_simple_floating_species_concentrations_store'],
                            'time': ['time_store']}
            },
            'amici_simple': {
                '_type': 'process',
                'address': 'local:amici',
                'config': {'model': {'model_source': model_fp}},
                'inputs': {
                    'floating_species_concentrations': ['amici_simple_floating_species_concentrations_store'],
                    'model_parameters': ['model_parameters_store'],
                    'time': ['time_store'],
                    'reactions': ['reactions_store']},
                'outputs': {
                    'floating_species_concentrations': ['amici_simple_floating_species_concentrations_store'],
                    'time': ['time_store']}
            },
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'emit': {
                        'copasi_simple_floating_species_concentrations': 'tree[float]',
                        'amici_simple_floating_species_concentrations': 'tree[float]',
                        'tellurium_simple_floating_species_concentrations': 'tree[float]',
                        'time': 'float'
                    }
                },
                'inputs': {
                    'copasi_simple_floating_species_concentrations': ['copasi_simple_floating_species_concentrations_store'],
                    'amici_simple_floating_species_concentrations': ['amici_simple_floating_species_concentrations_store'],
                    'tellurium_simple_floating_species_concentrations': ['tellurium_simple_floating_species_concentrations_store'],
                    'time': ['time_store']
                }
            },
            'tellurium_simple': {
                '_type': 'process',
                'address': 'local:tellurium',
                'config': {'model': {'model_source': model_fp}},
                'inputs': {'floating_species_concentrations': ['tellurium_simple_floating_species_concentrations_store'],
                           'model_parameters': ['model_parameters_store'],
                           'time': ['time_store'],
                           'reactions': ['reactions_store']},
                'outputs': {'floating_species_concentrations': ['tellurium_simple_floating_species_concentrations_store'],
                            'time': ['time_store']}}}

    # TODO: Do we need this?
    def inputs(self):
        return {}

    def outputs(self):
        return {'comparison_data': 'tree[any]'}

    def update(self, state):
        comp = self._generate_composition()
        results = self._run_composition(comp)
        output = {'comparison_data': results}
        return output

    def _generate_composition(self) -> Composite:
        return Composite(config={'state': self.document}, core=CORE)

    def _run_composition(self, comp: Composite) -> Dict:
        comp.run(self.duration)
        return comp.gather_results()


# comparison functions
def calculate_mse(a, b) -> int:
    return np.mean((a - b) ** 2)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, atol=None, rtol=None) -> bool:
    """Original methodology copied from biosimulations runutils."""
    if isinstance(arr1[0], np.float64):
        max1 = max(arr1)
        max2 = max(arr2)
        aTol = atol or max(1e-3, max1*1e-5, max2*1e-5)
        rTol = rtol or 1e-4
        return np.allclose(arr1, arr2, rtol=rTol, atol=aTol)

    for n in range(len(arr1)):
        if not compare_arrays(arr1[n], arr2[n]):
            return False
    return True


def run_simulator_comparison(
        simulator_data: Dict[str, np.ndarray],
        project_id: str,
        rtol: float = 1e-4,
        atol: Optional[float] = None,
        ground_truth: Optional[np.ndarray] = None,
        ) -> SimulatorComparison:
    """Run a simulator comparison accross n simulators, where n is the number of keys defined in `simulator_data`. Optionally,
       compare the individual simulators against a `ground_truth` if one is passed. Ground truth could represent any
       target data.

            Args:
                simulator_data:`Dict[str, np.ndarray]`: dict mapping of {SIMULATOR NAME: OUTPUT OF SIMULATOR NAME} for
                    each simulator datum in the comparison.
                project_id:`str`: project ID for comparison.
                rtol:`float`: relative tolerance for comparison.
                atol:`float`: absolute tolerance for comparison.
                ground_truth:`np.ndarray`: If passed, this value is compared against each simulator in simulator_data.keys().
                    Defaults to `None`.

            Returns:
                SimulatorComparison object mapping.
    """
    # TODO: Finish this!
    pass


def generate_comparison_matrix(
        outputs: List[np.ndarray],
        simulators: List[str],
        method: Union[str, any] = 'prox',
        rtol: float = None,
        atol: float = None,
        ground_truth: np.ndarray = None,
        matrix_id: str = None
        ) -> ComparisonMatrix:
    """Generate a Mean Squared Error comparison matrix of arr1 and arr2, indexed by simulators by default,
            or an AllClose Tolerance routine result if `method` is set to `prox`.

            Args:
                outputs: list of output arrays.
                simulators: list of simulator names.
                matrix_id: name/id of the comparison
                method: pass one of either: `mse` to perform a mean-squared error calculation
                    or `prox` to perform a pair-wise proximity tolerance test using `np.allclose(outputs[i], outputs[i+1])`.
                rtol:`float`: relative tolerance for comparison if `prox` is used.
                atol:`float`: absolute tolerance for comparison if `prox` is used.
                ground_truth: If passed, this value is compared against each simulator in simulators. Currently, this
                    field is agnostic to any verified/validated source, and we trust that the user has verified it. Defaults
                    to `None`.

            Returns:
                ComparisonMatrix object consisting of:
                    - `name`: the id of the matrix
                    - `data`: Pandas dataframe representing a comparison matrix where `i` and `j` are both indexed by the
                        simulators involved. The aforementioned simulators involved will also include the `ground_truth` value
                        within the indices if one is passed.
                    - `ground_truth`: Reference to the ground truth vals if used.
    """
    matrix_data = generate_matrix_data(outputs, simulators, method, rtol, atol, ground_truth)
    return ComparisonMatrix(name=matrix_id, data=matrix_data, ground_truth=ground_truth)


def generate_matrix_data(
    outputs: List[np.ndarray],
    simulators: List[str],
    method: Union[str, any] = 'prox',
    rtol: float = None,
    atol: float = None,
    ground_truth: np.ndarray = None
    ) -> pd.DataFrame:
    """Generate a Mean Squared Error comparison matrix of arr1 and arr2, indexed by simulators by default,
        or an AllClose Tolerance routine result if `method` is set to `prox`.

        Args:
            outputs: list of output arrays.
            simulators: list of simulator names.
            method: pass one of either: `mse` to perform a mean-squared error calculation
                or `prox` to perform a pair-wise proximity tolerance test using `np.allclose(outputs[i], outputs[i+1])`.
            rtol:`float`: relative tolerance for comparison if `prox` is used.
            atol:`float`: absolute tolerance for comparison if `prox` is used.
            ground_truth: If passed, this value is compared against each simulator in simulators. Currently, this
                field is agnostic to any verified/validated source, and we trust that the user has verified it. Defaults
                to `None`.

        Returns:
            Pandas dataframe representing a comparison matrix where `i` and `j` are both indexed by the
                simulators involved. The aforementioned simulators involved will also include the `ground_truth` value
                within the indices if one is passed.
    """

    # TODO: map arrs to simulators more tightly.
    if ground_truth is not None:
        simulators.append('ground_truth')
        outputs.append(ground_truth)

    use_tol_method = method.lower() == 'prox'
    matrix_dtype = float if not use_tol_method else bool
    mse_matrix = np.zeros((3, 3), dtype=matrix_dtype)

    # fill the matrices with the calculated values
    for i in range(len(simulators)):
        for j in range(i, len(simulators)):
            output_i = outputs[i]
            output_j = outputs[j]
            method_type = method.lower()

            result = calculate_mse(output_i, output_j) \
                if method_type == 'mse' else compare_arrays(output_i, output_j, rtol, atol) if use_tol_method else None
            assert result is not None, "You must pass a valid method argument value of either mse or tol"
            # mse_matrix[i, j] = calculate_mse(output_i, output_j) if not use_tol_method else compare_arrays(output_i, output_j, rtol, atol)

            mse_matrix[i, j] = result
            if i != j:
                mse_matrix[j, i] = mse_matrix[i, j]

    return pd.DataFrame(mse_matrix, index=simulators, columns=simulators)
