from typing import List, Tuple


def register_bigraph_module(items_to_register: List[Tuple[str, str]], core) -> None:
    for process_name, path in items_to_register:
        module_name, class_name = path.rsplit('.', 1)
        try:
            import_statement = f'src.{module_name}'

            module = __import__(
                 import_statement, fromlist=[class_name])

            # module = importlib.import_module(import_statement)

            # Get the class from the module
            bigraph_class = getattr(module, class_name)

            # Register the process
            core.process_registry.register(process_name, bigraph_class)
            print(f"{class_name} registered successfully.")
        except ImportError as e:
            print(f"{class_name} not available. Error: {e}")
