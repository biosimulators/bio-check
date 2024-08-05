# -- api models -- #
from typing import List, Optional, Any

from shared import BaseModel, Job


class PendingOmexJob(Job):
    job_id: str
    status: str
    timestamp: str
    path: str
    simulators: List[str]
    comparison_id: Optional[str] = None
    ground_truth_report_path: Optional[str] = None
    include_output: Optional[bool] = True
    rTol: Optional[float] = None
    aTol: Optional[float] = None
    selection_list: Optional[List[str]] = None


class PendingSbmlJob(Job):
    job_id: str
    status: str
    timestamp: str
    path: str
    start: int 
    end: int
    steps: int
    simulators: List[str]
    comparison_id: Optional[str] = None
    include_output: Optional[bool] = True
    rTol: Optional[float] = None
    aTol: Optional[float] = None
    selection_list: Optional[List[str]] = None


class OmexComparisonSubmission(PendingOmexJob):
    pass


class SbmlComparisonSubmission(PendingSbmlJob):
    pass


class Simulators(BaseModel):
    simulators: List[str]


# class UtcComparisonSubmission(PendingJob):
    # pass


class DbClientResponse(BaseModel):
    message: str
    db_type: str  # ie: 'mongo', 'postgres', etc
    timestamp: str


class UtcComparisonRequestParams(BaseModel):
    simulators: List[str] = ["amici", "copasi", "tellurium"]
    include_output: Optional[bool] = True
    comparison_id: Optional[str] = None


class UtcComparisonResult(BaseModel):
    content: Any


class Simulator(BaseModel):
    name: str
    version: Optional[str] = None


class CompatibleSimulators(BaseModel):
    file: str
    simulators: List[Simulator]



