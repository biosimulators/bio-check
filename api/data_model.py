# -- api models -- #
from typing import List, Optional, Any

from shared import BaseModel, Job


class PendingJob(Job):
    job_id: str
    status: str
    timestamp: str
    omex_path: str
    simulators: List[str]
    timestamp: str
    comparison_id: Optional[str] = None
    ground_truth_report_path: Optional[str] = None
    include_output: Optional[bool] = True


class UtcComparisonSubmission(PendingJob):
    pass


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



