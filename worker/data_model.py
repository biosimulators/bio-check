# -- worker models -- #
from dataclasses import dataclass, asdict
from typing import *

import numpy as np

from shared_worker import BaseClass


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
