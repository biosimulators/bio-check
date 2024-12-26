from typing import *

from process_bigraph import ProcessTypes


def register_module(items_to_register: List[Tuple[str, str]], core: ProcessTypes, verbose=False) -> None:
    for process_name, path in items_to_register:
        module_name, class_name = path.rsplit('.', 1)
        try:
            library = 'steps' if 'process' not in path else 'processes'
            import_statement = f'biosimulators_processes.{library}.{module_name}'

            module = __import__(
                 import_statement, fromlist=[class_name])

            # module = importlib.import_module(import_statement)

            # Get the class from the module
            bigraph_class = getattr(module, class_name)

            # Register the process
            core.process_registry.register(process_name, bigraph_class)
        except Exception as e:
            print(f"Cannot register {class_name}. Error:\n**\n{e}\n**")
            continue
