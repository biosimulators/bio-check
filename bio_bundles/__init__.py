import importlib
from typing import *

from bigraph_schema import TypeSystem
from process_bigraph import Composite, ProcessTypes

from biosimulators_processes.steps.viz import CompositionPlotter, Plotter2d
from biosimulators_processes.data_model.sed_data_model import MODEL_TYPE, UTC_CONFIG_TYPE
from bio_bundles.registry import register_module


# Define a list of processes to attempt to import and register
PROCESSES_TO_REGISTER = [
    ('cobra-process', 'cobra_process.CobraProcess'),
    ('copasi-process', 'copasi_process.CopasiProcess'),
    ('tellurium-process', 'tellurium_process.TelluriumProcess'),
    ('utc-amici', 'amici_process.UtcAmici'),
    ('utc-copasi', 'copasi_process.UtcCopasi'),
    ('utc-tellurium', 'tellurium_process.UtcTellurium')]


try:
    import smoldyn
    PROCESSES_TO_REGISTER.append(('smoldyn-process', 'smoldyn_process.SmoldynProcess'))
    PROCESSES_TO_REGISTER.append(('smoldyn-io-process', 'smoldyn_process.SmoldynIOProcess'))
except:
    print('Smoldyn is not properly installed in this environment and thus its process implementation cannot be registered. Please consult smoldyn documentation.')


STEPS_TO_REGISTER = [
    ('copasi-step', 'ode_simulation.CopasiStep'),
    ('tellurium-step', 'ode_simulation.TelluriumStep'),
    ('amici-step', 'ode_simulation.AmiciStep'),
    ('plotter', 'viz.CompositionPlotter'),
    ('plotter2d', 'viz.Plotter2d'),
    ('utc-comparator', 'comparator_step.UtcComparator'),
    ('smoldyn-step', 'bio_compose.SmoldynStep'),
    ('simularium-smoldyn-step', 'bio_compose.SimulariumSmoldynStep'),
    ('database-emitter', 'bio_compose.MongoDatabaseEmitter')
    # ('simple-neuron', 'neuron_network.SimpleNeuron'),
    # ('simple-neuron-network', 'neuron_network.SimpleNeuronNetwork')]
]

# core process registry implementation (unique to this package)
CORE = ProcessTypes()

# core type system implementation (unique to this package)
CORE.type_registry.register('sed_model', schema={'_type': MODEL_TYPE})
CORE.type_registry.register('utc_config', schema={'_type': UTC_CONFIG_TYPE})
register_module(PROCESSES_TO_REGISTER, CORE, verbose=False)
register_module(STEPS_TO_REGISTER, CORE, verbose=True)