import subprocess
import sys
import importlib

from shared.log_config import setup_logging


logger = setup_logging(__file__)


def format_dynamic_install(simulators: list[str], package_name: str = None) -> str:
    package = f"{package_name or 'biosimulator-processes'}["
    for i, sim in enumerate(simulators):
        if not i == len(simulators) - 1:
            package += sim + ","
        else:
            package += sim + "]"
    return package


def install_pypi_package(pypi_handle: str, verbose: bool = True) -> int:
    """
    Dynamically installs a given pypi package.

    :param pypi_handle: (`str`) name of the package to install. Anything that would be called with `pip install ...` INCLUDING optional extras if needed dep[a,b,...]
    :param verbose: (`bool`) whether to print progress confirmations
    """
    if verbose:
        print(f"Installing {pypi_handle}...")
    # run pip install
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pypi_handle])
        if verbose:
            print(f"{pypi_handle} installed successfully.")
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f">> {str(e)}")
        raise e


def install_request_dependencies(simulators: list[str]) -> int:
    handle = format_dynamic_install(simulators=simulators)
    return install_pypi_package(pypi_handle=handle, verbose=False)


