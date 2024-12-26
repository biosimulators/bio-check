# -- gateway models -- #
# -- worker models -- #
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import *
from typing import List, Optional, Any, Dict, Union

from dotenv import load_dotenv
from pydantic import Field, BaseModel as _BaseModel, ConfigDict
from fastapi.responses import FileResponse
import numpy as np


REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
DEV_ENV_PATH = os.path.join(REPO_ROOT, 'assets', 'dev', 'config', '.env')

load_dotenv(DEV_ENV_PATH)

DB_TYPE = "mongo"
DB_NAME = "service_requests"
BUCKET_NAME = os.getenv("BUCKET_NAME")


class BaseModel(_BaseModel):
    """Base Pydantic Model with custom app configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class BaseClass:
    """Base Python Dataclass multipurpose class with custom app configuration."""
    def to_dict(self):
        return asdict(self)


class SBMLSpeciesMapping(dict):
    pass


# PENDING JOBS:

class JobStatus(Enum):
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class DatabaseCollections(Enum):
    PENDING_JOBS = "PENDING_JOBS".lower()
    IN_PROGRESS_JOBS = "IN_PROGRESS_JOBS".lower()
    COMPLETED_JOBS = "COMPLETED_JOBS".lower()


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


# ReaddyRun config:
# {
#   "species_config": [
#     {
#       "name": "E",
#       "diffusion_constant": 10.0
#     },
#     {
#       "name": "S",
#       "diffusion_constant": 10.0
#     },
#     {
#       "name": "ES",
#       "diffusion_constant": 10.0
#     },
#     {
#       "name": "P",
#       "diffusion_constant": 10.0
#     }
#   ],
#   "reactions_config": [
#     {
#       "scheme": "fwd: E +(0.03) S -> ES",
#       "rate": 86.78638438
#     },
#     {
#       "scheme": "back: ES -> E +(0.03) S", "rate": 1.0
#     },
#     {
#       "scheme": "prod: ES -> E +(0.03) P", "rate": 1.0
#     }
#   ],
#   "particles_config": [
#     {
#       "name": "E",
#       "initial_positions": [
#          [-0.11010841,  0.01048227, -0.07514985],
#          [0.02715631, -0.03829782,  0.14395517],
#          [0.05522253, -0.11880506,  0.02222362]
#       ]
#     },
#     {
#       "name": "S",
#       "initial_positions": [
#          [-0.21010841,  0.21048227, -0.07514985],
#          [0.02715631, -0.03829782,  0.14395517],
#          [0.05522253, -0.11880506,  0.02222362]
#       ]
#     }
#   ],
#   "unit_system_config": {
#     "length_unit": "micrometer",
#     "time_unit": "second"
#   }
# }


class ReaddySpeciesConfig(BaseModel):
    name: str
    diffusion_constant: float


class ReaddyReactionConfig(BaseModel):
    scheme: str
    rate: float


class ReaddyParticleConfig(BaseModel):
    name: str
    initial_positions: List[List[float]]


class ReaddyRun(BaseModel):
    job_id: str
    timestamp: str
    status: str
    duration: float
    dt: float
    box_size: List[float]
    species_config: Union[Dict[str, float], List[ReaddySpeciesConfig]]
    particles_config: Union[Dict[str, List[List[float]]], List[ReaddyParticleConfig]]
    reactions_config: Union[Dict[str, float], List[ReaddyReactionConfig]]
    simulators: Optional[List[str]] = ["readdy"]
    unit_system_config: Optional[Dict[str, str]] = {"length_unit": "micrometer", "time_unit": "second"}
    reaction_handler: Optional[str] = "UncontrolledApproximation"


class UtcRun(ApiRun):
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


class FileJob(BaseModel, FileResponse):
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


class OutputData(BaseModel):
    content: Union[Any, SmoldynOutput]


# -- verification --

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


class BigraphRegistryAddresses(BaseModel):
    version: str
    registered_addresses: List[str]


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


@dataclass
class BiosimulationsReportOutput(BaseClass):
    dataset_label: str
    data: np.ndarray


@dataclass
class BiosimulationsRunOutputData(BaseClass):
    report_path: str
    data: List[BiosimulationsReportOutput]


# these are data model-style representation of the functions from output_generator:
@dataclass
class NodeSpec:
    address: str
    config: Dict[str, Any]
    inputs: Dict[str, Any]
    outputs: Dict[str, Any]
    _type: str
    name: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class StepNodeSpec(NodeSpec):
    _type: str = "step"


@dataclass
class ProcessNodeSpec(NodeSpec):
    _type: str = "process"


@dataclass
class CompositionSpec:
    """To be used as input to process_bigraph.Composition() like:

        spec = CompositionSpec(nodes=nodes, emitter_mode='ports')
        composite = Composition({'state': spec
    """
    nodes: List[NodeSpec]
    emitter_mode: str = "all"

    def get_spec(self):
        return {
            node_spec.name: node_spec
            for node_spec in self.nodes
        }
