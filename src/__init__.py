from dataclasses import dataclass, asdict

from builder import ProcessTypes
from bigraph_schema import TypeSystem

from src.utils import register_bigraph_module


REGISTRATION_DATA = [('ode-comparison', 'compare.OdeComparatorStep')]
CORE = ProcessTypes()
TYPE_SYSTEM = TypeSystem()

register_bigraph_module(REGISTRATION_DATA, CORE)

