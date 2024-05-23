from typing import *

import numpy as np
from pydantic import FilePath, Field

from src.data_model import BaseModel
from src.data_model.arguments import Simulator


class Result(BaseModel):
    name: str
    metadata: Optional[Dict[str, Union[str, List[str]]]] = Field(default=None)
    value: Optional[Dict] = Field(default=None)


class VerificationResult(Result):
    """
        Attributes:
            simulators:`List[Simulator]`: simulators used in the verification
            ground_truth:`Optional[np.ndarray]`: ground truth used in the comparison. Defaults to `None`.
    """
    simulators: List[Simulator]
    ground_truth: Optional[np.ndarray] = Field(default=None)


class Url(VerificationResult):
    pass


# polymorphic
class Plot(VerificationResult):
    data: List


class ResultFile(VerificationResult):
    location: FilePath


class HDF5File(ResultFile):
    pass


class CSVFile(ResultFile):
    pass


class SEDMLFile(ResultFile):
    pass


class SimulationRun(Result):
    simulator: Simulator
    simulation_id: str
    project_id: str
    status: str = Field(default="Unknown")








