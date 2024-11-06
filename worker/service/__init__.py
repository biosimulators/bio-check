from typing import List, Tuple

from process_bigraph import ProcessTypes


APP_PROCESS_REGISTRY = ProcessTypes()
IMPLEMENTATIONS = [
    ('output-generator', 'steps.OutputGenerator'),
    ('time-course-output-generator', 'steps.TimeCourseOutputGenerator'),
    ('smoldyn_step', 'steps.SmoldynStep'),
    ('simularium_smoldyn_step', 'steps.SimulariumSmoldynStep'),
    ('mongo-emitter', 'steps.MongoDatabaseEmitter')
]


def register_module(
        items_to_register: List[Tuple[str, str]],
        verbose=False
) -> None:
    for process_name, path in items_to_register:
        module_name, class_name = path.rsplit('.', 1)
        try:
            import_statement = f'worker.bigraph.{module_name}'

            module = __import__(
                 import_statement, fromlist=[class_name])

            # Get the class from the module
            bigraph_class = getattr(module, class_name)

            # Register the process
            APP_PROCESS_REGISTRY.process_registry.register(process_name, bigraph_class)
            print(f'Registered {process_name}') if verbose else None
        except Exception as e:
            print(f"Cannot register {class_name}. Error:\n**\n{e}\n**") if verbose else None
            continue


register_module(IMPLEMENTATIONS)


# APP_PROCESS_REGISTRY.process_registry.register('time-course-output-generator', TimeCourseOutputGenerator)
# APP_PROCESS_REGISTRY.process_registry.register('output-generator', OutputGenerator)

