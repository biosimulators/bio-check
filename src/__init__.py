from dataclasses import dataclass, asdict

from pydantic import BaseModel as _BaseModel, ConfigDict

from src.data_model.arguments import (
    ModelFile,
    OMEXArchive,
    SBMLFile,
    CellMLFile,
    SEDMLFile,
    SED2File,
    SEDMArchiveFile,
    BigraphCompositionFile,
    TimeCourseSimulationFile,
    Simulator,
    DefaultSimulator,
    Package,
    DefaultAmici,
    DefaultCopasi,
    DefaultTellurium,
    ComparisonMethod,
    MSEComparisonMethod,
    DefaultComparisonMethod,
    CustomComparisonMethod
)
from src.data_model.results import (
    VerificationResult,
    Url,
    Plot,
    Result,
    CSVFile,
    HDF5File,
    SEDMLFile,
    SimulationRun
)



@dataclass
class _BaseClass:
    def to_dict(self):
        return asdict(self)


class BaseModel(_BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
