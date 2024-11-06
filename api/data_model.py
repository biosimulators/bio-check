# -- api models -- #
from enum import Enum
from typing import List, Optional, Any, Dict, Union

from pydantic import Field
from fastapi.responses import FileResponse

from api.shared_api import BaseModel, Job, JobStatus


# PENDING JOBS:

class ApiRun(BaseModel):
    job_id: str
    timestamp: str
    status: str
    path: str
    simulators: List[str]


class SmoldynRun(ApiRun):
    duration: Optional[int] = None
    dt: Optional[float] = None
    simulators: List[str] = ["smoldyn"]


class UtcRun(ApiRun):
    start: int
    end: int
    steps: int


class VerificationRun(ApiRun):
    include_outputs: Optional[bool] = True
    selection_list: Optional[List[str]] = None
    # expected_results: Optional[str] = None
    comparison_id: Optional[str] = None
    rTol: Optional[float] = None
    aTol: Optional[float] = None


class OmexVerificationRun(VerificationRun):
    pass


class SbmlVerificationRun(VerificationRun):
    start: int
    end: int
    steps: int


# IN PROGRESS JOBS:

# TODO: Implement this:
class IncompleteJob(BaseModel):
    job_id: str
    timestamp: str
    status: str
    source: Optional[str] = None


class SmoldynJob(IncompleteJob):
    pass


# COMPLETED JOBS:

# TODO: parse results and make object specs
class ObservableData(BaseModel):
    observable_name: str
    mse: Dict[str, Any]   # Dict[str, float]]
    proximity: Dict[str, Any]   #  Dict[str, bool]]
    output_data: Dict[str, Union[List[float], str]]   #  List[float]]


class SimulatorRMSE(BaseModel):
    simulator: str
    rmse_scores: Dict[str, float]


class Output(BaseModel):
    job_id: str
    timestamp: str
    status: str
    source: Optional[str] = None


class SmoldynOutput(FileResponse):
    pass


class VerificationOutput(Output):
    """
    Return schema for get-verification-output.

    Parameters:
        job_id: str
        timestamp: str
        status: str --> may be COMPLETE, IN_PROGRESS, FAILED, or PENDING
        source: str
        requested_simulators: List[str]
        results: Optional[dict] = None TODO: parse this
    """
    requested_simulators: Optional[List[str]] = None
    results: Optional[Union[List[Union[ObservableData, SimulatorRMSE, Any]], Dict[str, Any]]] = None


class OutputData(BaseModel):
    content: Union[VerificationOutput, SmoldynOutput]


# -- verification --

class PendingOmexVerificationJob(Job):
    job_id: str
    status: str
    timestamp: str
    path: str
    simulators: List[str]
    comparison_id: Optional[str] = None
    expected_results: Optional[str] = None
    include_output: Optional[bool] = True
    rTol: Optional[float] = None
    aTol: Optional[float] = None
    selection_list: Optional[List[str]] = None


class PendingSbmlVerificationJob(Job):
    job_id: str
    status: str
    timestamp: str
    path: str
    start: int 
    end: int
    steps: int
    simulators: List[str]
    comparison_id: Optional[str] = None
    expected_results: Optional[str] = None
    include_output: Optional[bool] = True
    rTol: Optional[float] = None
    aTol: Optional[float] = None
    selection_list: Optional[List[str]] = None


# -- simulation execution --

class PendingSmoldynJob(Job):
    job_id: str 
    timestamp: str
    path: str
    status: str = "PENDING"
    duration: Optional[int] = None
    dt: Optional[float] = None
    # initial_species_counts: Optional[Dict] = None


class PendingUtcJob(Job):
    job_id: str
    timestamp: str
    status: str
    path: str
    start: int
    end: int
    steps: int
    simulator: str


# -- files --

class PendingSimulariumJob(Job):
    """jobid timestamp path filename box_size status"""
    job_id: str
    timestamp: str
    path: str
    filename: str
    box_size: float
    status: str = "PENDING"


# -- composition --

class CompositionNode(BaseModel):
    name: str = Field(default=None, examples=["fba-process"], description="Descriptive name of the composition node.")
    node_type: str = Field(examples=['<IMPLEMENTATION TYPE>'], description="Type name, usually either step or process.")
    address: str = Field(examples=['<ADDRESS PROTOCOL>:<ADDRESS ID>'], description="Node (process or step) implementation address within Bigraph schema via some sort of ProcessTypes implementation.")
    config: Optional[Any] = Field(
        default=None,
        examples=[{'<REQUIRED PARAMETER NAME>': '<REQUIRED PARAMETER VALUE>'}],
        description="A mapping of config_schema names to required values as per the given process bigraph step or process implementation."
    )
    inputs: Optional[Dict[str, List[str]]] = Field(
        default=None,
        examples=[{'<INPUT PORT NAME>': ['<INPUT PORT STORE NAME>']}],
        description="A mapping of input port (data) names and a list describing the path at which results for that data name are stored within the composite bigraph."
    )
    outputs: Optional[Dict[str, List[str]]] = Field(
        default=None,
        examples=[{'<OUTPUT PORT NAME>': ['<OUTPUT PORT STORE NAME>']}],
        description="A mapping of output port (data) names and a list describing the path at which results for that data name are stored within the composite bigraph."
    )


class AgentParameter(BaseModel):
    name: str
    radius: Optional[float]
    mass: Optional[float]
    density: Optional[float]


class AgentParameters(BaseModel):
    agents: List[AgentParameter]


# class CompositionSpecification(BaseModel):
#     composition_id: str = Field(default=None, examples=["ode-fba-species-a"], description="Unique composition ID.")
#     nodes: List[CompositionNode]

class CompositionSpecification(BaseModel):
    composition_id: str
    nodes: List[Any]


class PendingCompositionJob(BaseModel):
    job_id: str
    composition: Dict[str, Any]
    duration: int
    timestamp: str
    status: str = JobStatus.PENDING.value


class DbClientResponse(BaseModel):
    message: str
    db_type: str  # ie: 'mongo', 'postgres', etc
    timestamp: str


# -- data --

class _OutputData(BaseModel):
    content: Any


class UtcComparisonResult(_OutputData):
    pass


class Simulator(BaseModel):
    name: str
    version: Optional[str] = None


class CompatibleSimulators(BaseModel):
    file: str
    simulators: List[Simulator]



