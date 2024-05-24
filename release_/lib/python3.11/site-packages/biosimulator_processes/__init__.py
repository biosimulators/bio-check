from process_bigraph import process_registry


# Define a list of processes to attempt to import and register
processes_to_register = [
    ('cobra', 'cobra_process.CobraProcess'),
    ('copasi', 'copasi_process.CopasiProcess'),
    ('smoldyn', 'smoldyn_process.SmoldynProcess'),
    ('tellurium', 'tellurium_process.TelluriumProcess'),
]

for process_name, process_path in processes_to_register:
    module_name, class_name = process_path.rsplit('.', 1)
    try:
        # Dynamically import the module
        process_module = __import__(f'biosimulator_processes.{module_name}', fromlist=[class_name])
        # Get the class from the module
        process_class = getattr(process_module, class_name)

        # Register the process
        process_registry.register(process_name, process_class)
        print(f"{class_name} registered successfully.")
    except ImportError as e:
        print(f"{class_name} not available. Error: {e}")
