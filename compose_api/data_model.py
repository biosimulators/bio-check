# -- api models -- #
from enum import Enum
from typing import List, Optional, Any, Dict

from pydantic import Field

from shared import BaseModel, Job, JobStatus


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
    config: Dict[str, Any] = Field(
        examples=[{'<REQUIRED PARAMETER NAME>': '<REQUIRED PARAMETER VALUE>'}],
        description="A mapping of config_schema names to required values as per the given process bigraph step or process implementation."
    )
    inputs: Optional[Dict[str, List[str]]] = Field(
        examples=[{'<INPUT PORT NAME>': ['<INPUT PORT STORE NAME>']}],
        description="A mapping of input port (data) names and a list describing the path at which results for that data name are stored within the composite bigraph."
    )
    outputs: Optional[Dict[str, List[str]]] = Field(
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


class CompositionSpecification(BaseModel):
    composition_id: str = Field(default=None, examples=["ode-fba-species-a"], description="Unique composition ID.")
    nodes: List[CompositionNode]


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

class OutputData(BaseModel):
    content: Any


class UtcComparisonResult(OutputData):
    pass


class Simulator(BaseModel):
    name: str
    version: Optional[str] = None


class CompatibleSimulators(BaseModel):
    file: str
    simulators: List[Simulator]



