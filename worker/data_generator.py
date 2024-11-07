import logging
from tempfile import mkdtemp
from pprint import pformat
from typing import *

import libsbml
from process_bigraph import Composite, ProcessTypes

from log_config import setup_logging
from shared_worker import handle_exception
from compatible import COMPATIBLE_UTC_SIMULATORS
from io_worker import normalize_smoldyn_output_path_in_root, get_sbml_species_mapping

# logging TODO: implement this.
logger = logging.getLogger("biochecknet.worker.data_generator.log")
setup_logging(logger)

AMICI_ENABLED = True
COPASI_ENABLED = True
PYSCES_ENABLED = True
TELLURIUM_ENABLED = True
SMOLDYN_ENABLED = True

try:
    from amici import SbmlImporter, import_model_module, Model, runAmiciSimulation
except ImportError as e:
    AMICI_ENABLED = False
    logger.warn(str(e))
try:
    from basico import *
except ImportError as e:
    COPASI_ENABLED = False
    logger.warn(str(e))
try:
    import tellurium as te
except ImportError as e:
    TELLURIUM_ENABLED = False
    logger.warn(str(e))
try:
    from smoldyn import Simulation
except ImportError as e:
    SMOLDYN_ENABLED = False
    logger.warn(str(e))
try:
    import pysces
except ImportError as e:
    PYSCES_ENABLED = False
    logger.warn(str(e))


# class OutputGenerator(Step):
#     config_schema = {
#         'input_file': 'string',
#         'context': 'string',
#     }
#
#     def __init__(self, config=None, core=APP_PROCESS_REGISTRY):
#         super().__init__(config, core)
#         self.input_file = self.config['input_file']
#         self.context = self.config.get('context')
#         if self.context is None:
#             raise ValueError("context (i.e., simulator name) must be specified in this processes' config.")
#
#     @abstractmethod
#     def generate(self, parameters: Optional[Dict[str, Any]] = None):
#         """Abstract method for generating output data upon which to base analysis from based on its origin.
#
#         This can be used for logic of any scope.
#         NOTE: args and kwargs are not defined in this function, but rather should be defined by the
#         inheriting class' constructor: i,e; start_time, etc.
#
#         Kwargs relate only to the given simulator api you are working with.
#         """
#         pass
#
#     def initial_state(self):
#         # base class method
#         return {
#             'output_data': {}
#         }
#
#     def inputs(self):
#         return {
#             'parameters': 'tree[any]'
#         }
#
#     def outputs(self):
#         return {
#             'output_data': 'tree[any]'
#         }
#
#     def update(self, state):
#         parameters = state.get('parameters') if isinstance(state, dict) else {}
#         data = self.generate(parameters)
#         return {'output_data': data}
#
#
# class TimeCourseOutputGenerator(OutputGenerator):
#     # NOTE: we include defaults here as opposed to constructor for the purpose of deliberate declaration within .json state representation.
#     config_schema = {
#         # 'input_file': 'string',
#         # 'context': 'string',
#         'start_time': {
#             '_type': 'integer',
#             '_default': 0
#         },
#         'end_time': {
#             '_type': 'integer',
#             '_default': 10
#         },
#         'num_steps': {
#             '_type': 'integer',
#             '_default': 100
#         },
#     }
#
#     def __init__(self, config=None, core=APP_PROCESS_REGISTRY):
#         super().__init__(config, core)
#         if not self.input_file.endswith('.xml'):
#             raise ValueError('Input file must be a valid SBML (XML) file')
#
#         self.start_time = self.config.get('start_time')
#         self.end_time = self.config.get('end_time')
#         self.num_steps = self.config.get('num_steps')
#         self.species_mapping = get_sbml_species_mapping(self.input_file)
#
#     def initial_state(self):
#         # TODO: implement this
#         pass
#
#     def generate(self, parameters: Optional[Dict[str, Any]] = None):
#         # TODO: add kwargs (initial state specs) here
#         executor = SBML_EXECUTORS[self.context]
#         data = executor(self.input_file, self.start_time, self.end_time, self.num_steps)
#         return data


# register utc instance:
# CORE.process_registry.register('time-course-output-generator', TimeCourseOutputGenerator)
# register generic instance:
# CORE.process_registry.register('output-generator', OutputGenerator)


# -- functions related to generating time course output data (for verification and more) using the process-bigraph engine -- #

def node_spec(_type: str, address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any], name: str = None):
    spec = {
        '_type': _type,
        'address': address,
        'config': config,
        'inputs': inputs,
        'outputs': outputs
    }

    return {name: spec} if name else spec


def step_node_spec(address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any], name: str = None):
    return node_spec(name=name, _type="step", address=address, config=config, inputs=inputs, outputs=outputs)


def process_node_spec(address: str, config: Dict[str, Any], inputs: Dict[str, Any], outputs: Dict[str, Any], name: str = None):
    return node_spec(name=name, _type="process", address=address, config=config, inputs=inputs, outputs=outputs)


def time_course_node_spec(input_file: str, context: str, start_time: int, end_time: int, num_steps: int):
    config = {
        'input_file': input_file,
        'start_time': start_time,
        'end_time': end_time,
        'num_steps': num_steps,
        'context': context
    }
    return step_node_spec(
        address='local:time-course-output-generator',
        config=config,
        inputs={
            'parameters': [f'parameters_store_{context}']
        },
        outputs={
            'output_data': [f'output_data_store_{context}']
        }
    )


def generate_time_course_data(
        input_fp: str,
        start: int,
        end: int,
        steps: int,
        core: ProcessTypes,
        simulators: List[str] = None,
        parameters: Dict[str, Any] = None,
        expected_results_fp: str = None,
        out_dir: str = None
) -> Dict[str, Dict[str, List[float]]]:
    requested_sims = simulators or ["amici", "copasi", "pysces", "tellurium"]
    simulation_spec = {
        simulator: time_course_node_spec(
            input_file=input_fp,
            context=simulator,
            start_time=start,
            end_time=end,
            num_steps=steps
        ) for simulator in requested_sims
    }
    simulation = Composite({'state': simulation_spec, 'emitter': {'mode': 'all'}}, core=core)

    input_filename = input_fp.split("/")[-1].split(".")[0]
    if out_dir:
        simulation.save(
            filename=f'{input_filename}-initialization.json',
            outdir=out_dir
        )

    # TODO: is there a better way to do this? (interval of one? Is that symbolic more than anything?)
    if parameters:
        simulation.update(parameters, 1)
    else:
        simulation.run(1)

    if out_dir:
        simulation.save(
            filename=f'{input_filename}-update.json',
            outdir=out_dir
        )

    output_data = {}
    raw_data = simulation.gather_results()[('emitter',)]
    for data in raw_data:
        for data_key, data_value in data.items():
            if data_key.startswith('output_data_store_'):
                simulator = data_key.split('_')[-1]
                output_data[simulator] = data_value

    return output_data


# -- direct simulator API wrappers --

# TODO: should we return the actual data from memory, or that reflected in a smoldyn output txt file or both?
def run_smoldyn(model_fp: str, duration: int, dt: float = None) -> Dict[str, Union[str, Dict[str, Union[float, List[float]]]]]:
    """Run the simulation model found at `model_fp` for the duration
        specified therein if output_files are specified in the smoldyn model file and return the aforementioned output file
        or return a dictionary of an array of the `listmols` as well as `molcount` command outputs. NOTE: The model file is currently
        searched for this `output_files` value, and if it exists and not commented out, it will scan the root of the model_fp
        (usually where smoldyn output files are stored, which is the same dir as the model_fp) to retrieve the output file.

            Args:
                model_fp:`str`: path to the smoldyn configuration. Defaults to `None`.
                duration:`float`: duration in seconds to run the simulation for.
                dt:`float`: time step in seconds to run the simulation for. Defaults to None, which uses the built-in simulation dt.

        For the output, we should read the model file and search for "output_files" to start one of the lines.
        If it startswith that, then assume a return of the output txt file, if not: then assume a return from ram.
    """
    # search for output_files in model_fp TODO: optimize this
    use_file_output = False
    with open(model_fp, 'r') as f:
        model_content = [line.strip() for line in f.readlines()]
        for content in model_content:
            if content.startswith('output_files'):
                use_file_output = True
        f.close()

    output_data = {}
    simulation = Simulation.fromFile(model_fp)
    try:
        # case: there is no declaration of output_files in the smoldyn config file, or it is commented out
        if not use_file_output:
            # write molcounts to counts dataset at every timestep (shape=(n_timesteps, 1+n_species <-- one for time)): [timestep, countSpec1, countSpec2, ...]
            simulation.addOutputData('species_counts')
            simulation.addCommand(cmd='molcount species_counts', cmd_type='E')

            # write spatial output to molecules dataset
            simulation.addOutputData('molecules')
            simulation.addCommand(cmd='listmols molecules', cmd_type='E')

            # run simulation for specified time
            step_size = dt or simulation.dt
            simulation.run(duration, step_size, overwrite=True)

            species_count = simulation.count()['species']
            species_names: List[str] = []
            for index in range(species_count):
                species_name = simulation.getSpeciesName(index)
                if 'empty' not in species_name.lower():
                    species_names.append(species_name)

            molecule_output = simulation.getOutputData('molecules')
            counts_output = simulation.getOutputData('species_counts')
            for i, output_array in enumerate(counts_output):
                interval_data = {}
                for j, species_count in enumerate(output_array):
                    interval_data[species_names[j - 1]] = species_count
                counts_output.pop(i)
                counts_output.insert(i, interval_data)

            # return ram data (default dimensions)
            output_data = {'species_counts': counts_output, 'molecules': molecule_output}

        # case: output files are specified, and thus time parameters by which to capture/collect output
        else:
            # run simulation with default time params
            simulation.runSim()

            # change the output filename to a standardized 'modelout.txt' name
            working_dir = os.path.dirname(model_fp)
            results_fp = normalize_smoldyn_output_path_in_root(working_dir)

            # return output file
            output_data = {'results_file': results_fp}
    except:
        error = handle_exception("Run Smoldyn")
        logger.error(error)
        output_data = {'error': error}

    return output_data


def handle_sbml_exception() -> str:
    tb_str = traceback.format_exc()
    error_message = pformat(f"time-course-simulation-error:\n{tb_str}")
    return error_message


def run_sbml_pysces(sbml_fp: str, start, dur, steps):
    if PYSCES_ENABLED:
        # model compilation
        sbml_filename = sbml_fp.split('/')[-1]
        psc_filename = sbml_filename + '.psc'
        psc_fp = os.path.join(pysces.model_dir, psc_filename)

        # get output with mapping of internal species ids to external (shared) species names
        sbml_species_mapping = get_sbml_species_mapping(sbml_fp)
        obs_names = list(sbml_species_mapping.keys())
        obs_ids = list(sbml_species_mapping.values())

        # run the simulation with specified time params and get the data
        try:
            model = pysces.loadSBML(sbmlfile=sbml_fp, pscfile=psc_fp)
            model.sim_time = np.linspace(start, dur, steps + 1)
            model.Simulate(1)  # specify userinit=1 to directly use model.sim_time (t) rather than the default
            return {
                name: model.data_sim.getSimData(obs_id)[:, 1].tolist()
                for name, obs_id in sbml_species_mapping.items()
            }
        except:
            error_message = handle_sbml_exception()
            logger.error(error_message)
            return {"error": error_message}
    else:
        return OSError('Pysces is not properly installed in your environment.')


def run_sbml_tellurium(sbml_fp: str, start, dur, steps):
    result = None
    try:
        simulator = te.loadSBMLModel(sbml_fp)
        if start > 0:
            simulator.simulate(0, start)
        result = simulator.simulate(start, dur, steps + 1)
        species_mapping = get_sbml_species_mapping(sbml_fp)
        if result is not None:
            outputs = {}
            for colname in result.colnames:
                if 'time' not in colname:
                    for spec_name, spec_id in species_mapping.items():
                        if colname.replace("[", "").replace("]", "") == spec_id:
                            data = result[colname]
                            outputs[spec_name] = data.tolist()
            return outputs
        else:
            raise Exception('Tellurium: Could not generate results.')
    except:
        error_message = handle_sbml_exception()
        logger.error(error_message)
        return {"error": error_message}


def run_sbml_copasi(sbml_fp, start, dur, steps):
    try:
        t = np.linspace(start, dur, steps + 1)
        model = load_model(sbml_fp)
        specs = get_species(model=model).index.tolist()
        for spec in specs:
            if spec == "EmptySet" or "EmptySet" in spec:
                specs.remove(spec)
        tc = run_time_course(model=model, update_model=True, values=t)
        data = {spec: tc[spec].values.tolist() for spec in specs}
        return data
    except:
        error_message = handle_sbml_exception()
        logger.error(error_message)
        return {"error": error_message}


def run_sbml_amici(sbml_fp: str, start, dur, steps):
    try:
        sbml_reader = libsbml.SBMLReader()
        sbml_doc = sbml_reader.readSBML(sbml_fp)
        sbml_model_object = sbml_doc.getModel()
        sbml_importer = SbmlImporter(sbml_fp)
        model_id = sbml_fp.split('/')[-1].replace('.xml', '')
        model_output_dir = mkdtemp()
        sbml_importer.sbml2amici(
            model_id,
            model_output_dir,
            verbose=logging.INFO,
            observables=None,
            sigmas=None,
            constant_parameters=None
        )
        # model_output_dir = model_id  # mkdtemp()
        model_module = import_model_module(model_id, model_output_dir)
        amici_model_object: Model = model_module.getModel()
        floating_species_list = list(amici_model_object.getStateIds())
        floating_species_initial = list(amici_model_object.getInitialStates())
        sbml_species_ids = [spec.getName() for spec in sbml_model_object.getListOfSpecies()]
        t = np.linspace(start, dur, steps + 1)
        amici_model_object.setTimepoints(t)
        initial_state = dict(zip(floating_species_list, floating_species_initial))
        set_values = []
        for species_id, value in initial_state.items():
            set_values.append(value)
        amici_model_object.setInitialStates(set_values)
        sbml_species_mapping = get_sbml_species_mapping(sbml_fp)
        method = amici_model_object.getSolver()
        result_data = runAmiciSimulation(solver=method, model=amici_model_object)
        results = {}
        floating_results = dict(zip(
            sbml_species_ids,
            list(map(lambda x: result_data.by_id(x), floating_species_list))
        ))
        results = floating_results

        return results
    except:
        error_message = handle_sbml_exception()
        logger.error(error_message)
        return {"error": error_message}


# TODO: add vcell and masspy here
SBML_EXECUTORS = dict(zip(
    [data[0] for data in COMPATIBLE_UTC_SIMULATORS],
    [run_sbml_amici, run_sbml_copasi, run_sbml_pysces, run_sbml_tellurium]
))
