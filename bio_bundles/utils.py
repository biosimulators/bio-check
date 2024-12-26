import traceback
from pprint import pformat


def handle_exception(error_key: str = "bio-compose-error") -> str:
    tb_str = traceback.format_exc()
    error_message = pformat(f"{error_key}:\n{tb_str}")
    return error_message


def handle_sbml_exception() -> str:
    return handle_exception("time-course-simulation-error")

