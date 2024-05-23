from dataclasses import dataclass

from src.data_model import _BaseClass


@dataclass
class OutputAspectVerification(_BaseClass):
    aspect_type: str  # one of: 'names', 'values'. TODO: Add more
    is_verified: bool
    expected_data: any
    process_data: any
