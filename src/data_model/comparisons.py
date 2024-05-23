from typing import List, Tuple

import numpy as np

from src import BaseModel


class PairwiseComparison(BaseModel):
    edge: Tuple[np.ndarray, np.ndarray]
    value: bool


class SimulatorComparison:
    project_id: str
    data: List[PairwiseComparison]
