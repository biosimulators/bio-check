from typing import *

import numpy as np

from src.enter import OMEXArchive, SBMLFile, CellMLFile, EntryPointFile, SimulationFile, Simulator
from src.result import VerificationResult
from src.verification.compare import generate_comparison_matrix


def main_src():
    pass


def verify(
    in_file: EntryPointFile,
    simulation_file: SimulationFile,
    simulators: List[Simulator],
    ground_truth: np.ndarray = None,
    comparison_method: str = 'mse',
    output_type: str = 'csv'
) -> VerificationResult:
    # 1. iterate over each simulator in the simulators list and load each simulator with the in_file and append to a mapping of sims
    # 2. run the simulation method for each sim in the mapping from #1
    # 3. Map each output from #2 back to each simulator from the simulators list.
    # 4. Run these simulators and their respective results through generate_comparison_matrix
    # 5. Return a verification result (one of: url, plots, hdf5, csv, sedml
    pass


def verify_batch(files: List[EntryPointFile]):
    # TODO: iterate over files and args with verify
    pass
