from typing import *

import numpy as np
import pandas as pd

from server.handlers.output_data import generate_species_output


def generate_utc_species_comparison(omex_fp, out_dir, species_name, simulators):
    outputs = generate_species_output(omex_fp, out_dir, species_name)
    methods = ['mse', 'prox']
    return dict(zip(
        methods,
        list(map(
            lambda m: generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m).to_dict(),
            methods
        ))
    ))


def calculate_mse(a, b) -> float:
    return np.mean((a - b) ** 2)


def compare_arrays(arr1: np.ndarray, arr2: np.ndarray, atol=None, rtol=None) -> bool:
    """Original methodology copied from biosimulations runutils."""
    max1 = max(arr1)
    max2 = max(arr2)
    aTol = atol or max(1e-3, max1*1e-5, max2*1e-5)
    rTol = rtol or 1e-4
    return np.allclose(arr1, arr2, rtol=rTol, atol=aTol)


def generate_species_comparison_matrix(
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

    # TODO: implement the ground truth
    _simulators = simulators.copy()
    _outputs = outputs.copy()
    if ground_truth is not None:
        _simulators.append('ground_truth')
        _outputs.append(ground_truth)

    use_tol_method = method.lower() == 'prox'
    matrix_dtype = np.float64 if not use_tol_method else bool
    num_simulators = len(_simulators)
    mse_matrix = np.zeros((num_simulators, num_simulators), dtype=matrix_dtype)

    # fill the matrices with the calculated values
    for i in range(len(_simulators)):
        for j in range(i, len(_simulators)):
            output_i = _outputs[i]
            output_j = _outputs[j]
            method_type = method.lower()
            result = calculate_mse(output_i, output_j) if method_type == 'mse' else compare_arrays(arr1=output_i, arr2=output_j, rtol=rtol, atol=atol)

            mse_matrix[i, j] = result
            if i != j:
                mse_matrix[j, i] = mse_matrix[i, j]

    return pd.DataFrame(mse_matrix, index=_simulators, columns=_simulators)
