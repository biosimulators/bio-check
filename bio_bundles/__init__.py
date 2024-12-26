import importlib
from typing import *

from process_bigraph import ProcessTypes

from bio_bundles.data_model.sed import MODEL_TYPE, UTC_CONFIG_TYPE
from bio_bundles.registry import Registrar


STEP_IMPLEMENTATIONS = [
    ('output-generator', 'main.OutputGenerator'),
    ('time-course-output-generator', 'main.TimeCourseOutputGenerator'),
    ('smoldyn_step', 'main.SmoldynStep'),
    ('simularium_smoldyn_step', 'main.SimulariumSmoldynStep'),
    ('mongo-emitter', 'main.MongoDatabaseEmitter'),
    ('copasi-step', 'ode_simulation.CopasiStep'),
    ('tellurium-step', 'ode_simulation.TelluriumStep'),
    ('amici-step', 'ode_simulation.AmiciStep'),
    ('plotter', 'viz.CompositionPlotter'),
    ('plotter2d', 'viz.Plotter2d'),
    ('utc-comparator', 'comparator_step.UtcComparator'),
    ('smoldyn-step', 'bio_compose.SmoldynStep'),
    ('simularium-smoldyn-step', 'bio_compose.SimulariumSmoldynStep'),
    ('database-emitter', 'bio_compose.MongoDatabaseEmitter')
]


PROCESS_IMPLEMENTATIONS = [
    ('cobra-process', 'cobra_process.CobraProcess'),
    ('copasi-process', 'copasi_process.CopasiProcess'),
    ('tellurium-process', 'tellurium_process.TelluriumProcess'),
    ('utc-amici', 'amici_process.UtcAmici'),
    ('utc-copasi', 'copasi_process.UtcCopasi'),
    ('utc-tellurium', 'tellurium_process.UtcTellurium')]


try:
    import smoldyn
    PROCESS_IMPLEMENTATIONS.append(('smoldyn-process', 'smoldyn_process.SmoldynProcess'))
    PROCESS_IMPLEMENTATIONS.append(('smoldyn-io-process', 'smoldyn_process.SmoldynIOProcess'))
except:
    print('Smoldyn is not properly installed in this environment and thus its process implementation cannot be registered. Please consult smoldyn documentation.')


# process/implementation registrar
registrar = Registrar()

# register types
registrar.register_type("sed_model", MODEL_TYPE)
registrar.register_type("utc_config", UTC_CONFIG_TYPE)

# register implementations of steps and processes
registrar.register_initial_modules(
    items_to_register=STEP_IMPLEMENTATIONS + PROCESS_IMPLEMENTATIONS
)




