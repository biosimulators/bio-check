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


def install_dependency(pypi_name: str, verbose: bool = True):
    """
    Dynamically installs a given simulator.

    :param pypi_name: (`str`) name of the package to install. Potentially the same as `simulator_import`. Included for redundant purposes.
    :param verbose: (`bool`) whether to print progress confirmations
    """
    # install using pip in the current environment
    print(f"Installing {pypi_name}...") if verbose else None
    subprocess.check_call([sys.executable, "-m", "pip", "install", pypi_name])
    print(f"{pypi_name} installed successfully.") if verbose else None


def dynamic_install(process_address: str, verbose: bool = True):
    if process_address.startswith("local:"):
        process_address = process_address.split(":")[-1]

    install = f"biosimulator-processes[{process_address}]"
    install_dependency(install, verbose=verbose)


def install_process_dependencies(process_address: str, verbose: bool = True):
    from bsp import app_registrar
    if process_address.startswith("local:"):
        process_address = process_address.split(":")[-1]

    process_deps = app_registrar.implementation_dependencies[process_address]
    for dep in process_deps:
        install_dependency(dep, verbose=verbose)





