import logging
import sys


def start_logging(fname: str):
    logging.basicConfig(
        filename=fname,
        level=logging.ERROR,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def setup_logging(fname: str, return_all: bool = False):
    # Create a root logger
    root_logger = logging.getLogger(fname)
    root_logger.setLevel(logging.INFO)

    # Create a console handler
    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create a formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Add the formatter to the console handler
    console_handler.setFormatter(formatter)

    # Add the console handler to the root logger and uvicorn logger
    root_logger.addHandler(console_handler)

    return root_logger, console_handler if return_all else root_logger
