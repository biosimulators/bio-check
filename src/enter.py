from typing import *

from pydantic import Field

from src import BaseModel


# base file class
class Entrypoint(BaseModel):
    pass


class EntryPointFile(Entrypoint):
    file_path: str


# model file entrypoints
class OMEXArchive(EntryPointFile):
    pass


class ModelFile(EntryPointFile):
    pass


class SBMLFile(ModelFile):
    pass


class CellMLFile(ModelFile):
    pass


# simulation arg base
class SimulationFile(EntryPointFile):
    pass


# simulation file arg
class SEDMArchiveFile(SimulationFile):
    archive_fp: str = Field(default=None)


class SEDMLFile(SimulationFile):
    pass


class LEMSFile(SimulationFile):
    pass


class SED2File(SimulationFile):  # ensure this is a json file!
    pass


class BigraphCompositionFile(SimulationFile):  # ensure this also is a json file!
    pass


class TimeCourseSimulationFile(SimulationFile):
    """Generic sim file for TC sims"""
    pass


class CellMLFile(SimulationFile):
    pass


# simulator arg
class Simulator(EntryPoint):
    name: str
    version: str


class DefaultSimulator(Simulator):
    name: str
    version: str = Field(default='latest')


# TODO: These are just ode sims: add to this collection asap.
class DefaultTellurium(DefaultSimulator):
    name: str = Field(default='tellurium')


class DefaultCopasi(DefaultSimulator):
    name: str = Field(default='copasi')


class DefaultAmici(DefaultSimulator):
    name: str = Field(default='amici')


# comparison method arg
class ComparisonMethod(EntryPoint):
    method_id: str


class DefaultComparisonMethod(ComparisonMethod):
    method_id: str = Field(default='tolerance_proximity')  # representing the alg based on np.close()














