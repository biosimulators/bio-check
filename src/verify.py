from dataclasses import dataclass


@dataclass
class OutputAspectVerification:
    aspect_type: str  # one of: 'names', 'values'. TODO: Add more
    is_verified: bool
    expected_data: any
    process_data: any
