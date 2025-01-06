import subprocess
import sys
import importlib


def install_simulators(simulators: list[str], verbose: bool = True):
    """
    Dynamically installs required simulator libraries.

    :param simulators: (`list[str]`) list of simulator libraries to install
    :param verbose: (`bool`) whether to print progress confirmations
    """
    for sim in simulators:
        try:
            # Check if the simulator is already installed
            importlib.import_module(sim)
            print(f"{sim} is already installed.") if verbose else None
        except ImportError:
            # Install using pip in the current environment
            print(f"Installing {sim}...") if verbose else None
            subprocess.check_call([sys.executable, "-m", "pip", "install", sim])
            print(f"{sim} installed successfully.") if verbose else None

