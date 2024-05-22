from typing import *

from pydantic import FilePath, Field

from src import BaseModel
from src.enter import Simulator


class Result(BaseModel):
    name: str
    metadata: Optional[Dict[str, Union[str, List[str]]]] = None


class Url(Result):
    pass


# polymorphic
class Plot(Result):
    data: List


class ResultFile(Result):
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






