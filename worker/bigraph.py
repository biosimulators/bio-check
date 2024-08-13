import os

from process_bigraph import Step
from process_bigraph.composite import ProcessTypes
try:
    import smoldyn as sm
    from smoldyn._smoldyn import MolecState
except:
    raise ImportError(
        '\nPLEASE NOTE: Smoldyn is not correctly installed on your system which prevents you from ' 
        'using the SmoldynProcess. Please refer to the README for further information '
        'on installing Smoldyn.'
    )


class SmoldynStep(Step):
    pass


class SimulariumStep(Step):
    pass


# register processes

CORE = ProcessTypes()
REGISTERED_PROCESSES = [
    ('smoldyn_step', SmoldynStep),
    ('simularium_step', SimulariumStep)
]
for process_name, process_class in REGISTERED_PROCESSES:
    try:
        CORE.process_registry.register(process_name, process_class)
    except Exception as e:
        print(f'{process_name} could not be registered because {str(e)}')

