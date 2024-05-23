from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from pydantic import Field

from src import BaseModel


class PairwiseComparison(BaseModel):
    edge: Tuple[np.ndarray, np.ndarray]
    value: bool


class SimulatorComparison(BaseModel):
    project_id: str
    data: List[PairwiseComparison]


class ComparisonMatrix(BaseModel):
    data: pd.DataFrame
    name: Optional[str] = Field(default="Unknown")
    ground_truth: Optional[np.ndarray] = Field(default=None)

