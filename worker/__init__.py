from typing import List, Tuple

from process_bigraph import ProcessTypes


APP_PROCESS_REGISTRY = ProcessTypes()
IMPLEMENTATIONS = [
    ('output-generator', 'OutputGenerator'),
    ('time-course-output-generator', 'TimeCourseOutputGenerator')
]


def register_module(
        items_to_register: List[Tuple[str, str]],
        core: ProcessTypes = APP_PROCESS_REGISTRY,
        verbose=False
) -> None:
    for process_name, class_name in items_to_register:
        try:
            import_statement = f'worker.output_generator'

            module = __import__(
                 import_statement, fromlist=[class_name])

            # Get the class from the module
            bigraph_class = getattr(module, class_name)

            # Register the process
            core.process_registry.register(process_name, bigraph_class)
            print(f'Registered {process_name}') if verbose else None
        except Exception as e:
            print(f"Cannot register {class_name}. Error:\n**\n{e}\n**")
            continue


register_module(IMPLEMENTATIONS)


# APP_PROCESS_REGISTRY.process_registry.register('time-course-output-generator', TimeCourseOutputGenerator)
# APP_PROCESS_REGISTRY.process_registry.register('output-generator', OutputGenerator)

