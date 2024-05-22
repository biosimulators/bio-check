from typing import *

from pydantic import FilePath

from src import BaseModel


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






