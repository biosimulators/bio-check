from typing import *
from types import FunctionType
from abc import ABC, abstractmethod

import numpy as np
from pydantic import Field, field_validator

from src import BaseModel
from src.verification.compare import pairwise_comparison


# TODO: seperate this into a lib for clarity

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


class ExperimentType(BaseModel):
    _type: str  # ie: ode, spatial, logical


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
class ComparisonMethod(EntryPoint, ABC):
    method_id: str

    @abstractmethod
    def run_comparison(self, arr1: np.ndarray, arr2: np.ndarray) -> bool:
        pass


class DefaultComparisonMethod(ComparisonMethod):
    method_id: str = Field(default='tolerance_proximity')  # representing the alg based on np.close()
    rtol: float = Field(default=1e-4)
    atol: float = Field(default=None)


    def run_comparison(self, arr1: np.ndarray, arr2: np.ndarray, ground_truth: np.ndarray, rtol=None, atol=None) -> dict[tuple, bool]:
        """Implementation of Comparison Method's inherited abstract method which compares two arrays using
             np.allclose() and rtol, atol if no `ground_truth` is provided, otherwise generate a pairwise comparison
             which includes the comparison of `ground_truth` to arr1 and arr2

             Args:
                 arr1 (np.array): first array
                 arr2 (np.array): second array
                 ground_truth (optional[list[np.ndarray]]): list of ground truth values by which to compare arr1 and arr2.
                     Defaults to `None`.
                 **kwargs (kwargs): keyword arguments passed to Comparison Method which include: rtol: float, atol: float.

            Returns:
                a dict where the keys are the comparison edges and the values are the result of the pairwise comparison for the given edge key.
        """
    def _run_comparison(self, arr1: np.ndarray, arr2: np.ndarray, ground_truth: np.ndarray = None, **kwargs) -> bool:

        if type(arr1[0]) == np.float64:
            max1 = max(arr1)
            max2 = max(arr2)
            atol = self.atol or kwargs.get('atol') or max(1e-3, max1 * 1e-5, max2 * 1e-5)
            return np.allclose(arr1, arr2, rtol=self.rtol, atol=atol)
        for n in range(len(arr1)):
            if not compare_arrays(arr1[n], arr2[n]):
                return False
        return True

    def compare_arr_to_truth(self, arr: np.ndarray, ground_truth: np.ndarray) -> ODEProcessIntervalComparison:
        pass


# TODO: update this more closely with the doc
class CustomComparisonMethod(ComparisonMethod):
    """An interface for custom comparison methods that can be consumed by the Rest API.
        The user must implement the run_comparison method.
        A user can pass a mapping of {'method_id': , 'run_comparison'...}
    """
    method_id: str  # must be defined to construct

    @abstractmethod
    def run_comparison(self, arr1: np.ndarray, arr2: np.ndarray) -> bool:
        pass














