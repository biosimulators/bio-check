# TODO: Implement this
from typing import List, Optional

from numpy import ndarray
from pandas import DataFrame

from src.compare import generate_comparison_matrix


def main(
    outputs: List[ndarray],
    simulators: List[str],
    method: str = 'prox',
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
    ground_truth: Optional[ndarray] = None
    ) -> DataFrame:
    pass


