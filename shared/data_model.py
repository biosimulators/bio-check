# -- gateway models -- #
# -- worker models -- #
import os
from dataclasses import dataclass, asdict
from enum import Enum
from typing import *

from dotenv import load_dotenv
from pydantic import Field, BaseModel as _BaseModel, ConfigDict
from fastapi.responses import FileResponse
import numpy as np


PROJECT_ROOT_PATH = os.path.dirname(os.path.dirname(__file__))
DEV_ENV_PATH = os.path.join(PROJECT_ROOT_PATH, 'shared', '.env')

load_dotenv(DEV_ENV_PATH)

DB_TYPE = os.getenv("DB_TYPE", "mongo")
DB_NAME = os.getenv("DB_NAME")
BUCKET_NAME = os.getenv("BUCKET_NAME")
JOB_COLLECTION_NAME = os.getenv("JOB_COLLECTION_NAME", "jobs")


class BaseModel(_BaseModel):
    """Base Pydantic Model with custom app configuration"""
    model_config = ConfigDict(arbitrary_types_allowed=True)


@dataclass
class BaseClass:
    """Base Python Dataclass multipurpose class with custom app configuration."""
    def to_dict(self):
        return asdict(self)


@dataclass
class WriteResponse(BaseClass):
    result: int

    @property
    def status(self):
        return "success" if self.result == 0 else "failed"


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


# IN PROGRESS JOBS:

# TODO: Implement this:
class IncompleteJob(BaseModel):
    job_id: str
    timestamp: str
    status: str
    source: Optional[str] = None


# -- spatial (readdy/smoldyn) --


class AgentParameter(BaseModel):
    name: str
    radius: Optional[float]
    mass: Optional[float]
    density: Optional[float]


class AgentParameters(BaseModel):
    agents: List[AgentParameter]


class SmoldynOutput(FileResponse):
    pass


class SmoldynRun(ApiRun):
    duration: Optional[int] = None
    dt: Optional[float] = None
    simulators: List[str] = ["smoldyn"]



class DbClientResponse(BaseModel):
    message: str
    db_type: str  # ie: 'mongo', 'postgres', etc
    timestamp: str


# -- process-bigraph specs -- TODO: implement this!

class BigraphRegistryAddresses(BaseModel):
    version: str
    registered_addresses: List[str]


class DataStorePath(str):
    pass


@dataclass
class DataStore(BaseClass):
    paths: List[DataStorePath | str] | str

    def __post_init__(self):
        if isinstance(self.paths, str):
            self.paths = [DataStorePath(self.paths)]
        else:
            paths = {
                path_i: path for path_i, path in enumerate(self.paths)
            }
            for i, path in paths.items():
                if isinstance(path, str):
                    paths[i] = DataStorePath(path)
            self.paths = list(paths.values())


@dataclass
class PortStore(BaseClass):
    name: str  # outermost keys under "inputs"
    store: DataStore | List[str]  # ie: ["concentrations_store"]

    def __post_init__(self):
        if isinstance(self.store, list):
            self.store = DataStore(self.store)


@dataclass
class CompositionNode(BaseClass):
    name: str
    _type: str
    address: str
    config: Dict[str, Any]
    inputs: Dict[str, List[str]]
    outputs: Optional[Dict[str, List[str]]] = None

    def to_dict(self):
        rep = super().to_dict()
        rep.pop("name")
        if not self.outputs:
            rep.pop("outputs")
        return rep


@dataclass
class CompositionSpec(BaseClass):
    """To be used as input to process_bigraph.Composition() like:

        spec = CompositionSpec(nodes=nodes, emitter_mode='ports')
        composite = Composition({'state': spec
    """
    nodes: List[CompositionNode]
    job_id: str
    emitter_mode: str = "all"

    @property
    def spec(self):
        return {
            node_spec.name: node_spec.to_dict()
            for node_spec in self.nodes
        }


@dataclass
class CompositionRun(BaseClass):
    job_id: str
    last_updated: str
    simulators: List[str]
    duration: int
    spec: Dict[str, Any]
    status: str = "PENDING"
    results: Dict[str, Any] = None


@dataclass
class OutputData(BaseClass):
    job_id: str
    status: str
    last_updated: str
    results: Dict
