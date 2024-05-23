from typing import *
from types import FunctionType, ModuleType
from abc import ABC, abstractmethod
import subprocess
import sys

import numpy as np
from pydantic import Field, field_validator
from biosimulators_utils.combine.io import CombineArchiveReader

from src import BaseModel
from src.service import BiosimulationsRestService
from src.verification.compare import generate_comparison_matrix


# TODO: seperate this into a lib for clarity

# base file class
class EntryPoint(BaseModel):
    pass


class EntryPointFile(EntryPoint):
    file_path: str


# model file entrypoints
class OMEXArchive(EntryPointFile):
    """
    If called, simply run the omex archive against the api.

        Attributes:
            file_path:`str`: Path to the OMEX archive file.
            out_dir:`str`: Path to the dir in which the archive will be upacked.
            project_id:`str`: TODO: Finish this and map it to the rest call.
    """
    out_dir: Optional[str] = Field(default=None)
    project_id: Optional[str] = Field(default=None)

    def get_model_file(self):
        return BiosimulationsRestService.get_model_file(project_id=self.project_id)


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
            name:`str`: the name of the package
            version:`str`: the version as specified in a given package index.
            module_name:`Optional[str]`: if specified, this refers to the package itself and enables this
                class a.k.a `Package` to use its `module` method, which effectively wraps the python `__import__` builtin
                such that it takes in an `import_statement` as an argument, which is
                equivalent to an `import ...` statement within a script and then,
                for package(particularly simulator) tool modules can be imported and used directly from a workflow.
    """
    name: str
    version: str = Field(default='latest')
    module_name: Optional[str] = Field(default=None)

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

    def module(self, import_statement: str = None) -> ModuleType:
        """module:`method`:(import_statement: `str` = None) -> `ModuleType`: if `module_name` is specified, this
            refers to how the package would be declared in a normal Python `import` statement. `import statement`
            refers to an specific nesting of import, otherwise the base pacakge will be imported.
        """
        statement = import_statement or self.module_name
        return __import__(statement)


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
    """Generic simulator instance to be used for any simulator package whose origin is PyPI. TODO: Expand this.

        Attributes:
            name:`str`: the name of the package
            version:`str`: the version as specified in the Python package index.
            module_name:`Optional[str]`: if specified, this refers to the package itself and enables this
                class a.k.a `Package` to use its `module` method, which effectively wraps the python `__import__` builtin
                such that it takes in an `import_statement` as an argument, which is
                equivalent to an `import ...` statement within a script and then,
                for package(particularly simulator) tool modules can be imported and used directly from a workflow.
            install_version:`method`:(version: `str`) -> `bool`: uninstalls the currently installed version and
                installs that which is spec'd in the `version` arg.
            module:`method`:(import_statement: `str` = None) -> `ModuleType`: if `module_name` is specified, this
                refers to how the package would be declared in a normal Python `import` statement. `import statement`
                refers to an specific nesting of import, otherwise the base pacakge will be imported.
    """
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
class ComparisonMethod(EntryPoint):
    method_id: str

    def run_comparison(self, **kwargs) -> pd.DataFrame:
        """Run a pairwise comparison and generate a dataframe representation of the output. Simply wraps
            the `src.verification.compare.generate_comparison_matrix(**kwargs)` function.

            Keyword Args:
                outputs – list of output arrays.
                simulators – list of simulator names.
                method – pass one of either: mse to perform a mean-squared error calculation or prox to perform a pair-wise proximity
                    tolerance test using np. allclose(outputs[i], outputs[i+1]).
                rtol – float: relative tolerance for comparison if prox is used.
                atol – float: absolute tolerance for comparison if prox is used.
                ground_truth – If passed, this value is compared against each simulator in simulators. Currently,
                    this field is agnostic to any verified/ validated source, and we trust that the user has verified it. Defaults to None.
        """
        return generate_comparison_matrix(**kwargs, method=self.method_id)


class MSEComparisonMethod(ComparisonMethod):
    method_id: str = Field(default='mse')


class ProximityComparisonMethod(ComparisonMethod):
    method_id: str = Field(default='prox')  # representing the alg based on np.close()
    rtol: float = Field(default=1e-4)
    atol: float = Field(default=None)


class DefaultComparisonMethod(ProximityComparisonMethod):
    """For now, we will use this as the default comparison method."""
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














