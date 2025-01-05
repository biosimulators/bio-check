from bio_bundles import registrar
from bio_bundles.data_model import BaseClass
from bio_bundles.registry import Registrar
from bio_bundles import registrar as app_registrar
from dataclasses import dataclass
from typing import *

from process_bigraph import Composite


@dataclass
class CompositionComponent(BaseClass):
    name: str

    def to_dict(self):
        node_dict = super().to_dict()
        node_dict.pop("name")
        return node_dict


@dataclass
class CompositionPort(CompositionComponent):
    store: List[str]


@dataclass
class CompositionNode(CompositionComponent):
    _type: str
    address: str
    config: Dict[str, Any]
    inputs: Dict[str, List[str]]
    outputs: Dict[str, List[str]]


class CompositionDocument(dict):
    pass


class CompositionBuilder:
    nodes: List[CompositionNode]
    document: CompositionDocument

    def __init__(self,
                 registrar: Optional[Registrar] = None,
                 nodes: Optional[List[CompositionNode]] = None):
        self.registrar = registrar or app_registrar
        self.nodes = nodes
        self.document = self.to_document()

    def to_document(self) -> CompositionDocument:
        import warnings

        doc = {}
        if len(self.nodes):
            for node in self.nodes:
                doc[node.name] = node.to_dict()
        else:
            warnings.warn("There are no nodes in the composition")

        return CompositionDocument(doc)

    def generate_composite(self) -> Composite:
        doc = self.to_document()
        core = self.registrar.core

        return Composite({
            "state": doc,
            "emitter": {"mode": "all"}
        }, core=core)

    def add_node(self, node: CompositionNode):
        self.nodes.append(node)
        self.document = self.to_document()

    def remove_node(self, node_name: str):
        if node_name not in list(self.document.keys()):
            raise ValueError(f"{node_name} not found in document.")
        else:
            for node in self.nodes:
                if node.name == node_name:
                    self.nodes.remove(node)
            self.document.pop(node_name)


def test():
    qaoa_spec = CompositionNode(
        name="qaoa",
        _type="process",
        address="local:qaoa",
        config={
            "n_variables": 5
        },
        inputs={
            "n_nodes": ["n_nodes_store"],
            "adjacentcy_matrix": ["adjacentcy_matrix_store"]
        },
        outputs={
            "n_nodes": ["n_nodes_store"],
            "bitstring": ["bitstring_store"]
        }
    )

    builder = CompositionBuilder(registrar, [qaoa_spec])
    doc = builder.to_document()

    doc['emitter'] = {
        "_type": "step",
        "address": "local:ram-emitter",
        "config": {
            "emit": {
                "n_nodes": "integer",
                "bitstring": "list[integer]"
            }
        },
        "inputs": {
            'n_nodes': ['n_nodes_store'],
            'bitstring': ['bitstring_store']
        }
    }
