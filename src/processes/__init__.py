from builder import ProcessTypes
from bigraph_schema import TypeSystem

from src.utils import register_bigraph_module


REGISTRATION_DATA = [
    ('ode-comparison', 'processes.comparison_process.OdeComparisonStep'),
    ('tellurium-step', 'processes.tellurium_process.TelluriumStep'),
    ('tellurium-process', 'processes.tellurium_process.TelluriumProcess'),
    ('amici-process', 'processes.amici_process.AmiciProcess'),
]
CORE = ProcessTypes()
TYPE_SYSTEM = TypeSystem()


register_bigraph_module(REGISTRATION_DATA, CORE)
