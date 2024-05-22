from typing import *
from types import FunctionType
from abc import ABC, abstractmethod
import subprocess
import sys

import numpy as np
from pydantic import Field, field_validator

from src import BaseModel


# TODO: seperate this into a lib for clarity

# base file class
class EntryPoint(BaseModel):
    pass


class EntryPointFile(EntryPoint):
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


class Package(EntryPoint):
    """Entrypoint implementation for any reference to a software package to be used at runtime.

        Attributes:
            name:`str`: the name of the package"""
    name: str
    version: str = Field(default='latest')

    def install_version(self, version: str) -> bool:
        """Removes currently installed version of """
        package = self.name + "==" + version
        print('Installing ' + package)
        try:
            # remove currently installed version
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", package])

            # install specified version
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

            return True
        except Exception as e:
            print(f'Installation failed:\n{e}')

            return False


# simulator and packages
class SimulatorTool(Package):
    """TODO: Somehow dynamically install the specified version if anything other than 'latest'."""
    pass


class PypiPackage(Package):
    """TODO: Make this python specific."""
    pass


class PypiSimulator(PypiPackage):
    pass


class Simulator(PypiSimulator):
    """Generic simulator instance to be used for any simulator package whose origin is PyPI. TODO: Expand this."""
    pass


class DefaultSimulator(PypiSimulator):
    """For now, we will default to using PyPI.
       TODO: resolve this to fit multiple package indices other than just pypi.
    """
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














