from typing import *

import numpy as np
from pydantic import FilePath, Field

from src import BaseModel
from src.enter import Simulator


class Result(BaseModel):
    name: str
    metadata: Optional[Dict[str, Union[str, List[str]]]] = None


class VerificationResult(Result):
    simulators: List[Simulator]
    ground_truth: np.ndarray = Field(default=None)


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








