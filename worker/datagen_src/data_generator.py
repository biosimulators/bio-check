import logging
from importlib import import_module
from tempfile import mkdtemp
import uuid
from pprint import pformat
from typing import *
from abc import abstractmethod
from logging import warn, Logger
from uuid import uuid4

import libsbml
from biosimulators_utils.config import Config
from kisao import AlgorithmSubstitutionPolicy
from process_bigraph import Step, Process
from process_bigraph.composite import Emitter, ProcessTypes, Composite
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from simulariumio import InputFileData, UnitData, DisplayData, DISPLAY_TYPE
from simulariumio.smoldyn import SmoldynData

from log_config import setup_logging
from shared_worker import handle_exception
from compatible import COMPATIBLE_UTC_SIMULATORS
from io_worker import normalize_smoldyn_output_path_in_root, get_sbml_species_mapping, read_report_outputs, read_h5_reports, make_dir
from simularium_utils import calculate_agent_radius, translate_data_object, write_simularium_file
from data_model import BiosimulationsRunOutputData

# logging TODO: implement this.
logger: Logger = logging.getLogger("biochecknet.worker.data_generator.log")
setup_logging(logger)

AMICI_ENABLED = True
COPASI_ENABLED = True
PYSCES_ENABLED = True
TELLURIUM_ENABLED = True
SMOLDYN_ENABLED = True
READDY_ENABLED = True

try:
    from amici import SbmlImporter, import_model_module, Model, runAmiciSimulation
except ImportError as e:
    AMICI_ENABLED = False
    logger.warning(str(e))
try:
    from basico import *
except ImportError as e:
    COPASI_ENABLED = False
    logger.warning(str(e))
try:
    import tellurium as te
except ImportError as e:
    TELLURIUM_ENABLED = False
    logger.warning(str(e))
try:
    from smoldyn import Simulation
    from smoldyn._smoldyn import MolecState
except ImportError as e:
    SMOLDYN_ENABLED = False
    logger.warning(str(e))
try:
    import readdy
except ImportError as e:
    READDY_ENABLED = False
    logger.warning(str(e))
try:
    import pysces
except ImportError as e:
    PYSCES_ENABLED = False
    logger.warning(str(e))

HISTORY_INDEXES = [
    'data.time',
    [('experiment_id', ASCENDING),
     ('data.time', ASCENDING),
     ('_id', ASCENDING)],
]
CONFIGURATION_INDEXES = [
    'experiment_id',
]
SECRETS_PATH = 'secrets.json'


# -- functions related to generating time course output data (for verification and more) using the process-bigraph engine -- #

class NodeSpec(dict):
    def __init__(self, _type: str, address: str, config: Dict[str, Any], inputs: Dict[str, List[str]], outputs: Dict[str, List[str]], name: str = None):
        super().__init__()
        self._type = _type
        self.address = address
        self.config = config
        self.inputs = inputs
        self.outputs = outputs
        self.name = name


def node_spec(_type: str, address: str, config: Dict[str, Any], inputs: Dict[str, List[str]], outputs: Dict[str, List[str]], name: str = None) -> Dict[str, Any]:
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
        core=None,
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
    if not out_dir:
        out_dir = mkdtemp()

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

    # return output_data
    import json
    with open(f'{out_dir}/{input_filename}-update.json', 'r') as f:
        state_spec = json.load(f)

    return {'output_data': output_data, 'state': state_spec}


def generate_composition_result_data(
        state_spec: Dict[str, Any],
        duration: int = None,
        core: ProcessTypes = None,
        out_dir: str = None
) -> Dict[str, Union[List[Dict[str, Any]], Dict[str, Any]]]:
    simulation = Composite({'state': state_spec, 'emitter': {'mode': 'all'}}, core=core)
    if duration is None:
        duration = 10
    simulation.run(duration)

    results = simulation.gather_results()[('emitter',)]

    import json
    if out_dir is None:
        out_dir = mkdtemp()
    with open(f'{out_dir}/update.json', 'r') as f:
        state_spec = json.load(f)

    return {'results': results, 'state': state_spec}


# -- direct simulator API wrappers -- #

def run_readdy(
        box_size: List[float],
        species_config: List[Dict[str, float]],   # {SPECIES_NAME: DIFFUSION_CONSTANT}  ie: {'E': 10.}
        reactions_config: List[Dict[str, float]],  # {REACTION_SCHEME: REACTION RATE}  ie: {"fwd: E +(0.03) S -> ES": 86.551}
        particles_config: List[Dict[str, Union[List[float], np.ndarray]]],  # {PARTICLE_NAME: INITIAL_POSITIONS_ARRAY}  ie: {'E': np.random.random(size=(n_particles_e, 3)) * edge_length - .5*edge_length}
        dt: float,
        duration: float,
        unit_system_config: Dict[str, str] = None
) -> Dict[str, str]:
    output = {}
    if READDY_ENABLED:
        # establish reaction network system
        unit_system = unit_system_config or {"length_unit": "micrometer", "time_unit": "second"}
        system = readdy.ReactionDiffusionSystem(
            box_size=box_size,
            unit_system=unit_system
        )

        # add species via spec
        species_names = []
        for config in species_config:
            species_name = config["name"]
            species_difc = config["diffusion_constant"]
            species_names.append(species_name)
            system.add_species(species_name, diffusion_constant=float(species_difc))

        # add reactions via spec
        for config in reactions_config:
            reaction_scheme = config["scheme"]
            reaction_rate = config["rate"]
            system.reactions.add(reaction_scheme, rate=float(reaction_rate))

        # configure simulation outputs
        simulation = system.simulation(kernel="CPU")
        simulation.output_file = "out.h5"
        simulation.reaction_handler = "UncontrolledApproximation"

        # set initial particle state and configure observations
        for config in particles_config:
            particle_name = config["name"]
            particle_positions = config["initial_positions"]
            if not isinstance(particle_positions, np.ndarray):
                particle_positions = np.array(particle_positions)
            simulation.add_particles(particle_name, particle_positions)
        simulation.observe.number_of_particles(
            stride=1,
            types=list(set(species_names))
        )

        # run simulation for given time parameters
        n_steps = int(float(duration) / dt)
        simulation.run(n_steps=n_steps, timestep=dt)
        output = {"results_file": simulation.output_file}
    else:
        error = handle_exception("Run Readdy")
        logger.error(error)
        output = {'error': error}

    return output


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


def run_sbml_pysces(sbml_fp: str, start: int, dur: int, steps: int) -> Dict[str, Union[List[float], str]]:
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
        # NOTE: the below model load works only in pysces 1.2.2 which is not available on conda via mac. TODO: fix this.
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


def run_sbml_tellurium(sbml_fp: str, start: int, dur: int, steps: int) -> Dict[str, Union[List[float], str]]:
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


def run_sbml_copasi(sbml_fp: str, start: int, dur: int, steps: int) -> Dict[str, Union[List[float], str]]:
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


def run_sbml_amici(sbml_fp: str, start: int, dur: int, steps: int) -> Dict[str, Union[List[float], str]]:
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
            list(map(
                lambda x: result_data.by_id(x),
                floating_species_list
            ))
        ))
        results = floating_results
        return {
            key: val.tolist() if isinstance(val, np.ndarray) else val
            for key, val in results.items()
        }
    except:
        error_message = handle_sbml_exception()
        logger.error(error_message)
        return {"error": error_message}


# TODO: add vcell and masspy here
SBML_EXECUTORS = dict(zip(
    [data[0] for data in COMPATIBLE_UTC_SIMULATORS],
    [run_sbml_amici, run_sbml_copasi, run_sbml_pysces, run_sbml_tellurium]
))


# -- formatted observables data -- #

def sbml_output_stack(spec_name: str, output):
    # 2. in output_stack: return {simname: output}
    stack = {}
    for simulator_name in output.keys():
        spec_data = output[simulator_name].get(spec_name)
        if isinstance(spec_data, str):
            data = None
        else:
            data = spec_data

        stack[simulator_name] = data

    return stack


def get_output_stack(spec_name: str, outputs):
    return sbml_output_stack(spec_name=spec_name, output=outputs)


def _get_report_output_stack(outputs: dict, spec_id: str):
    output_stack = []
    for sim_name in outputs.keys():
        sim_data = outputs[sim_name]['data']
        for data_index, data in enumerate(sim_data):
            data_id = data['dataset_label']
            if data_id == spec_id:
                # print(spec_id, data_id)
                output_stack.append(sim_data[data_index]['data'])
            else:
                pass
    return np.stack(output_stack)


def _generate_biosimulator_utc_outputs(omex_fp: str, output_root_dir: str, simulators: List[str] = None, alg_policy="same_framework") -> Dict:
    """Generate the outputs of the standard UTC simulators Copasi, Tellurium, and Amici from the
        biosimulators interface (exec_sedml_docs_in_combine_archive).
    """
    make_dir(output_root_dir)

    output_data = {}
    sims = simulators or ['amici', 'copasi', 'tellurium']
    sim_config = Config(
        LOG=False,
        ALGORITHM_SUBSTITUTION_POLICY=AlgorithmSubstitutionPolicy[alg_policy.upper()],
        VERBOSE=False)
    for sim in sims:
        sim_output_dir = os.path.join(output_root_dir, f'{sim}_outputs')
        make_dir(sim_output_dir)
        try:
            module = import_module(name=f'biosimulators_{sim}.core')
            exec_func = getattr(module, 'exec_sedml_docs_in_combine_archive')
            sim_output_dir = os.path.join(output_root_dir, f'{sim}_outputs')
            if not os.path.exists(sim_output_dir):
                os.mkdir(sim_output_dir)
            # execute simulator-specific simulation
            exec_func(archive_filename=omex_fp, out_dir=sim_output_dir, config=sim_config)
            report_path = os.path.join(sim_output_dir, 'reports.h5')

            sim_data = read_report_outputs(report_path)
            data = sim_data.to_dict() if isinstance(sim_data, BiosimulationsRunOutputData) else sim_data
            output_data[sim] = data
        except Exception as e:
            import traceback
            tb_str = traceback.format_exc()
            error_message = (
                f"An unexpected error occurred while processing your request:\n"
                f"Error Type: {type(e).__name__}\n"
                f"Error Details: {str(e)}\n"
                f"Traceback:\n{tb_str}"
            )
            output_data[sim] = error_message

    return output_data


def generate_biosimulator_utc_outputs(omex_fp: str, output_root_dir: str, simulators: list[str] = None, alg_policy="same_framework") -> dict:
    """Generate the outputs of the standard UTC simulators Copasi, Tellurium, and Amici from the
        biosimulators interface (exec_sedml_docs_in_combine_archive).
    """
    make_dir(output_root_dir)

    output_data = {}
    sims = simulators or ['amici', 'copasi', 'tellurium']  # , 'pysces']
    sim_config = Config(
        LOG=False,
        ALGORITHM_SUBSTITUTION_POLICY=AlgorithmSubstitutionPolicy[alg_policy.upper()],
        VERBOSE=False)
    for sim in sims:
        sim_output_dir = os.path.join(output_root_dir, f'{sim}_outputs')
        make_dir(sim_output_dir)
        try:
            module = import_module(name=f'biosimulators_{sim}.core')
            exec_func = getattr(module, 'exec_sedml_docs_in_combine_archive')
            sim_output_dir = os.path.join(output_root_dir, f'{sim}_outputs')
            if not os.path.exists(sim_output_dir):
                os.mkdir(sim_output_dir)

            # execute simulator-specific simulation
            exec_func(
                archive_filename=omex_fp,
                out_dir=sim_output_dir,
                config=sim_config if not sim == "pysces" else None
            )
            report_path = os.path.join(sim_output_dir, 'reports.h5')

            sim_data = read_h5_reports(report_path)
            data = sim_data.to_dict() if isinstance(sim_data, BiosimulationsRunOutputData) else sim_data
            output_data[sim] = data
        except:
            import traceback
            tb_str = traceback.format_exc()
            error_message = (
                f"Traceback:\n{tb_str}"
            )
            output_data[sim] = {'error': error_message}

    return output_data


def generate_sbml_utc_outputs(sbml_fp: str, start: int, dur: int, steps: int, simulators: list[str] = None, truth: str = None) -> dict:
    # TODO: add VCELL and pysces here
    output = {}
    sbml_species_ids = list(get_sbml_species_mapping(sbml_fp).keys())
    simulators = simulators or ['amici', 'copasi', 'tellurium', 'pysces']
    all_output_ids = []
    for simulator in simulators:
        results = {}
        simulator = simulator.lower()
        simulation_executor = SBML_EXECUTORS[simulator]
        sim_result = simulation_executor(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)

        # case: simulation execution was successful
        if "error" not in sim_result.keys():
            # add to all shared names
            all_output_ids.append(list(sim_result.keys()))

            # iterate over sbml_species_ids to index output data
            for species_id in sbml_species_ids:
                if species_id in sim_result.keys():
                    output_vals = sim_result[species_id]
                    if isinstance(output_vals, np.ndarray):
                        output_vals = output_vals.tolist()
                    results[species_id] = output_vals
        else:
            # case: simulation had an error
            results = sim_result

        # set the simulator output
        output[simulator] = results

    # get the commonly shared output ids
    final_output = {}
    shared_output_ids = min(all_output_ids)
    for simulator_name in output.keys():
        sim_data = {}
        for spec_id in output[simulator_name].keys():
            if spec_id in shared_output_ids:
                sim_data[spec_id] = output[simulator_name][spec_id]
            elif spec_id == "error":
                sim_data["error"] = output[simulator_name][spec_id]

        final_output[simulator_name] = sim_data

    # handle expected outputs
    if truth is not None:
        final_output['ground_truth'] = {}
        report_results = read_report_outputs(truth)
        report_data = report_results.to_dict()['data'] if isinstance(report_results, BiosimulationsRunOutputData) else {}
        for datum in report_data:
            spec_name = datum['dataset_label']
            if not spec_name.lower() == 'time':
                spec_data = datum['data']
                final_output['ground_truth'][spec_name] = spec_data

    return final_output


# -- process-bigraph implementations -- #

class MongoDatabaseEmitter(Emitter):
    client_dict: Dict[int, MongoClient] = {}
    config_schema = {
        'connection_uri': 'string',
        'experiment_id': 'maybe[string]',
        'emit_limit': {
            '_type': 'integer',
            '_default': 4000000
        },
        'database': 'maybe[string]'
    }

    @classmethod
    def create_indexes(cls, table: Any, columns: List[Any]) -> None:
        """Create the listed column indexes for the given DB table."""
        for column in columns:
            table.create_index(column)

    def __init__(self, config, core) -> None:
        """Config may have 'host' and 'database' items. The config passed is expected to be:

                {'experiment_id':,
                 'emit_limit':,
                 'embed_path':}

                TODO: Automate this process for the user in builder
        """
        super().__init__(config)
        self.core = core
        self.experiment_id = self.config.get('experiment_id', str(uuid.uuid4()))
        # In the worst case, `breakdown_data` can underestimate the size of
        # data by a factor of 4: len(str(0)) == 1 but 0 is a 4-byte int.
        # Use 4 MB as the breakdown limit to stay under MongoDB's 16 MB limit.
        self.emit_limit = self.config['emit_limit']

        # create new MongoClient per OS process
        curr_pid = os.getpid()
        if curr_pid not in MongoDatabaseEmitter.client_dict:
            MongoDatabaseEmitter.client_dict[curr_pid] = MongoClient(
                config['connection_uri'])
        self.client: MongoClient = MongoDatabaseEmitter.client_dict[curr_pid]

        # extract objects from current mongo client instance
        self.db: Database = getattr(self.client, self.config.get('database', 'simulations'))
        self.history_collection: Collection = getattr(self.db, 'history')
        self.configuration: Collection = getattr(self.db, 'configuration')

        # create column indexes for the given collection objects
        self.create_indexes(self.history_collection, HISTORY_INDEXES)
        self.create_indexes(self.configuration, CONFIGURATION_INDEXES)

        self.fallback_serializer = make_fallback_serializer_function(self.core)

    def query(self, query):
        return self.history_collection.find_one(query)

    def history(self):
        return [v for v in self.history_collection.find()]

    def flush_history(self):
        for v in self.history():
            self.history_collection.delete_one(v)

    def update(self, inputs):
        self.history_collection.insert_one(inputs)
        return {}


# -- simulators -- #

class SmoldynStep(Step):
    config_schema = {
        'model_filepath': 'string',
        'animate': {
            '_type': 'boolean',
            '_default': False
        },
        'duration': 'maybe[integer]',
        'dt': 'maybe[float]',
        'initial_species_counts': 'maybe[tree[float]]',
        'initial_mol_position': 'maybe[list[float]]',  # of particles/molecules
        'initial_mol_state': 'maybe[integer]',

        # TODO: Add a more nuanced way to describe and configure dynamic difcs given species interaction patterns
    }

    def __init__(self, config, core):
        """A new instance of `SmoldynProcess` based on the `config` that is passed. The schema for the config to be passed in
            this object's constructor is as follows:

            config_schema = {
                'model_filepath': 'string',  <-- analogous to python `str`
                'animate': 'bool'  <-- of type `bigraph_schema.base_types.bool`

            # TODO: It would be nice to have classes associated with this.
        """
        super().__init__(config=config, core=core)

        # specify the model fp for clarity
        self.model_filepath = self.config.get('model_filepath')

        # enforce model filepath passing
        if not self.model_filepath:
            raise ValueError(
                '''
                    The Process configuration requires a Smoldyn model filepath to be passed.
                    Please specify a 'model_filepath' in your instance configuration.
                '''
            )

        # initialize the simulator from a Smoldyn MinE.txt file.
        self.simulation: Simulation = Simulation.fromFile(self.model_filepath)

        # set default starting position of molecules/particles (assume all)
        self.initial_mol_position = self.config.get('initial_mol_position', [0.0, 0.0, 0.0])
        self.initial_mol_state = self.config.get('initial_mol_state', 0)

        # get a list of the simulation species
        species_count = self.simulation.count()['species']
        counts = self.config.get('initial_species_counts')
        self.initial_species_counts = counts
        self.species_names: List[str] = []
        for index in range(species_count):
            species_name = self.simulation.getSpeciesName(index)
            if 'empty' not in species_name.lower():
                self.species_names.append(species_name)

        self.initial_species_state = {}
        self.initial_mol_state = {}
        initial_mol_counts = {spec_name: self.simulation.getMoleculeCount(spec_name, MolecState.all) for spec_name in self.species_names}
        for species_name, count in initial_mol_counts.items():
            self.initial_species_state[species_name] = count
            for _ in range(count):
                self.initial_mol_state[str(uuid4())] = {
                    'coordinates': self.initial_mol_position,
                    'species_id': species_name,
                    'state': self.initial_mol_state
                }

        # sort for logistical mapping to species names (i.e: ['a', 'b', c'] == ['0', '1', '2']
        self.species_names.sort()

        # make species counts of molecules dataset for output
        self.simulation.addOutputData('species_counts')
        # write molcounts to counts dataset at every timestep (shape=(n_timesteps, 1+n_species <-- one for time)): [timestep, countSpec1, countSpec2, ...]
        self.simulation.addCommand(cmd='molcount species_counts', cmd_type='E')

        # make molecules dataset (molecule information) for output
        self.simulation.addOutputData('molecules')
        # write coords to dataset at every timestep (shape=(n_output_molecules, 7)): seven being [timestep, smol_id(species), mol_state, x, y, z, mol_serial_num]
        self.simulation.addCommand(cmd='listmols molecules', cmd_type='E')

        # initialize the molecule ids based on the species names. We need this value to properly emit the schema, which expects a single value from this to be a str(int)
        # the format for molecule_ids is expected to be: 'speciesId_moleculeNumber'
        self.molecule_ids = list(self.initial_mol_state.keys())

        # get the simulation boundaries, which in the case of Smoldyn denote the physical boundaries
        # TODO: add a verification method to ensure that the boundaries do not change on the next step...
        self.boundaries: Dict[str, List[float]] = dict(zip(['low', 'high'], self.simulation.getBoundaries()))

        # create a re-usable counts and molecules type to be used by both inputs and outputs
        self.counts_type = {
            species_name: 'integer'
            for species_name in self.species_names
        }

        self.output_port_schema = {
            'species_counts': {
                species_name: 'integer'
                for species_name in self.species_names
            },
            'molecules': 'tree[string]',  # self.molecules_type
            'results_file': 'string'
        }

        # set time if applicable
        self.duration = self.config.get('duration')
        self.dt = self.config.get('dt', self.simulation.dt)

        # set graphics (defaults to False)
        if self.config['animate']:
            self.simulation.addGraphics('opengl_better')

        self._specs = [None for _ in self.species_names]
        self._vals = dict(zip(self.species_names, [[] for _ in self.species_names]))

    # def initial_state(self):
    #     return {
    #         'species_counts': self.initial_species_state,
    #         'molecules': self.initial_mol_state
    #     }

    def inputs(self):
        # schema = self.output_port_schema.copy()
        # schema.pop('results_file')
        # return schema
        return {}

    def outputs(self):
        return self.output_port_schema

    def update(self, inputs) -> Dict:
        # reset the molecules, distribute the mols according to self.boundarieså
        # for name in self.species_names:
        #     self.set_uniform(
        #         species_name=name,
        #         count=inputs['species_counts'][name],
        #         kill_mol=False
        #     )

        # run the simulation for a given interval if specified, otherwise use builtin time
        if self.duration is not None:
            self.simulation.run(stop=self.duration, dt=self.simulation.dt, overwrite=True)
        else:
            self.simulation.runSim()

        # get the counts data, clear the buffer
        counts_data = self.simulation.getOutputData('species_counts')

        # get the final counts for the update
        final_count = counts_data[-1]
        # remove the timestep from the list
        final_count.pop(0)

        # get the data based on the commands added in the constructor, clear the buffer
        molecules_data = self.simulation.getOutputData('molecules')

        # create an empty simulation state mirroring that which is specified in the schema
        simulation_state = {
            'species_counts': {},
            'molecules': {}
        }

        # get and populate the species counts
        for index, name in enumerate(self.species_names):
            simulation_state['species_counts'][name] = counts_data[index]
            # input_counts = simulatio['species_counts'][name]
            # simulation_state['species_counts'][name] = int(final_count[index]) - input_counts

        # clear the list of known molecule ids and update the list of known molecule ids (convert to an intstring)
        # self.molecule_ids.clear()
        # for molecule in molecules_data:
            # self.molecule_ids.append(str(uuid4()))

        # get and populate the output molecules
        for i, single_mol_data in enumerate(molecules_data):
            mol_species_index = int(single_mol_data[1]) - 1
            mol_id = str(uuid4())
            simulation_state['molecules'][mol_id] = {
                'coordinates': single_mol_data[3:6],
                'species_id': self.species_names[mol_species_index],
                'state': str(int(single_mol_data[2]))
            }

        # mols = []
        # for index, mol_id in enumerate(self.molecule_ids):
        #     single_molecule_data = molecules_data[index]
        #     single_molecule_species_index = int(single_molecule_data[1]) - 1
        #     mols.append(single_molecule_species_index)
        #     simulation_state['molecules'][mol_id] = {
        #         'coordinates': single_molecule_data[3:6],
        #         'species_id': self.species_names[single_molecule_species_index],
        #         'state': str(int(single_molecule_data[2]))
        #     }

        # TODO -- post processing to get effective rates

        # TODO: adjust this for a more dynamic dir struct
        model_dir = os.path.dirname(self.model_filepath)
        for f in os.listdir(model_dir):
            if f.endswith('.txt') and 'out' in f:
                simulation_state['results_file'] = os.path.join(model_dir, f)

        return simulation_state

    def set_uniform(
            self,
            species_name: str,
            count: int,
            kill_mol: bool = True
    ) -> None:
        """Add a distribution of molecules to the solution in
            the simulation memory given a higher and lower bound x,y coordinate. Smoldyn assumes
            a global boundary versus individual species boundaries. Kills the molecule before dist if true.

            TODO: If pymunk expands the species compartment, account for
                  expanding `highpos` and `lowpos`. This method should be used within the body/logic of
                  the `update` class method.

            Args:
                species_name:`str`: name of the given molecule.
                count:`int`: number of molecules of the given `species_name` to add.
                kill_mol:`bool`: kills the molecule based on the `name` argument, which effectively
                    removes the molecule from simulation memory.
        """
        # kill the mol, effectively resetting it
        if kill_mol:
            self.simulation.runCommand(f'killmol {species_name}')

        # TODO: eventually allow for an expanding boundary ie in the configuration parameters (pymunk?), which is defies the methodology of smoldyn

        # redistribute the molecule according to the bounds
        self.simulation.addSolutionMolecules(
            species=species_name,
            number=count,
            highpos=self.boundaries['high'],
            lowpos=self.boundaries['low']
        )


class SimulariumSmoldynStep(Step):
    """
        agent_data should have the following structure:

        {species_name(type):
            {display_type: DISPLAY_TYPE.<REQUIRED SHAPE>,
             (mass: `float` AND density: `float`) OR (radius: `float`)

    """
    config_schema = {
        'output_dest': 'string',
        'box_size': 'float',  # as per simulariumio
        'spatial_units': {
            '_default': 'nm',
            '_type': 'string'
        },
        'temporal_units': {
            '_default': 'ns',
            '_type': 'string'
        },
        'translate_output': {
            '_default': True,
            '_type': 'boolean'
        },
        'write_json': {
            '_default': True,
            '_type': 'boolean'
        },
        'run_validation': {
            '_default': True,
            '_type': 'boolean'
        },
        'file_save_name': 'maybe[string]',
        'translation_magnitude': 'maybe[float]',
        'meta_data': 'maybe[tree[string]]',
        'agent_display_parameters': 'maybe[tree[string]]'  # as per biosim simularium
    }

    def __init__(self, config, core):
        super().__init__(config=config, core=core)

        # io params
        self.output_dest = self.config['output_dest']
        self.write_json = self.config['write_json']
        self.filename = self.config.get('file_save_name')

        # display params
        self.box_size = self.config['box_size']
        self.translate_output = self.config['translate_output']
        self.translation_magnitude = self.config.get('translation_magnitude')
        self.agent_display_parameters = self.config.get('agent_display_parameters', {})

        # units params
        self.spatial_units = self.config['spatial_units']
        self.temporal_units = self.config['temporal_units']

        # info params
        self.meta_data = self.config.get('meta_data')
        self.run_validation = self.config['run_validation']

    def inputs(self):
        return {'results_file': 'string', 'species_names': 'list[string]'}

    def outputs(self):
        return {'simularium_file': 'string'}

    def update(self, inputs):
        # get job params
        in_file = inputs['results_file']
        file_data = InputFileData(in_file)

        # get species data for display data
        species_names = inputs['species_names']

        # generate simulariumio Smoldyn Data TODO: should display data be gen for each species type or n number of instances of that type?
        display_data = self._generate_display_data(species_names)
        io_data = SmoldynData(
            smoldyn_file=file_data,
            spatial_units=UnitData(self.spatial_units),
            time_units=UnitData(self.temporal_units),
            display_data=display_data,
            meta_data=self.meta_data,
            center=True
        )

        # translate reflections if needed
        if self.translate_output:
            io_data = translate_data_object(data=io_data, box_size=self.box_size, translation_magnitude=self.translation_magnitude)
        # write data to simularium file
        if self.filename is None:
            self.filename = in_file.split('/')[-1].replace('.', '') + "-simulation"

        save_path = os.path.join(self.output_dest, self.filename)
        write_simularium_file(data=io_data, simularium_fp=save_path, json=self.write_json, validate=self.run_validation)
        result = {'simularium_file': save_path + '.simularium'}

        return result

    def _generate_display_data(self, species_names) -> Dict | None:
        # user is specifying display data for agents
        if isinstance(self.agent_display_parameters, dict) and len(self.agent_display_parameters.keys()) > 0:
            display_data = {}
            for name in species_names:
                display_params = self.agent_display_parameters[name]

                # handle agent radius
                radius_param = display_params.get('radius')

                # user has passed a mass and density for a given agent
                if radius_param is None:
                    radius_param = calculate_agent_radius(m=display_params['mass'], rho=display_params['density'])

                # make kwargs for display data
                display_data_kwargs = {
                    'name': name,
                    'display_type': DISPLAY_TYPE[display_params['display_type']],
                    'radius': radius_param
                }

                # check if self.agent_params as been passed as a mapping of species_name: {species_mass: , species_shape: }
                display_data[name] = DisplayData(**display_data_kwargs)

            return display_data

        return None


# -- Output data generators: -- #

class OutputGenerator(Step):
    config_schema = {
        'input_file': 'string',
        'context': 'string',
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.input_file = self.config['input_file']
        self.context = self.config.get('context')
        if self.context is None:
            raise ValueError("context (i.e., simulator name) must be specified in this processes' config.")

    @abstractmethod
    def generate(self, parameters: Optional[Dict[str, Any]] = None):
        """Abstract method for generating output data upon which to base analysis from based on its origin.

        This can be used for logic of any scope.
        NOTE: args and kwargs are not defined in this function, but rather should be defined by the
        inheriting class' constructor: i,e; start_time, etc.

        Kwargs relate only to the given simulator api you are working with.
        """
        pass

    def initial_state(self):
        # base class method
        return {
            'output_data': {}
        }

    def inputs(self):
        return {
            'parameters': 'tree[any]'
        }

    def outputs(self):
        return {
            'output_data': 'tree[any]'
        }

    def update(self, state):
        parameters = state.get('parameters') if isinstance(state, dict) else {}
        data = self.generate(parameters)
        return {'output_data': data}


class TimeCourseOutputGenerator(OutputGenerator):
    # NOTE: we include defaults here as opposed to constructor for the purpose of deliberate declaration within .json state representation.
    config_schema = {
        # 'input_file': 'string',
        # 'context': 'string',
        'start_time': {
            '_type': 'integer',
            '_default': 0
        },
        'end_time': {
            '_type': 'integer',
            '_default': 10
        },
        'num_steps': {
            '_type': 'integer',
            '_default': 100
        },
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        if not self.input_file.endswith('.xml'):
            raise ValueError('Input file must be a valid SBML (XML) file')

        self.start_time = self.config.get('start_time')
        self.end_time = self.config.get('end_time')
        self.num_steps = self.config.get('num_steps')
        self.species_mapping = get_sbml_species_mapping(self.input_file)

    def initial_state(self):
        # TODO: implement this
        pass

    def generate(self, parameters: Optional[Dict[str, Any]] = None):
        # TODO: add kwargs (initial state specs) here
        executor = SBML_EXECUTORS[self.context]
        data = executor(self.input_file, self.start_time, self.end_time, self.num_steps)

        return data


# -- process implementation utils -- #

def generate_simularium_file(
        input_fp: str,
        dest_dir: str,
        box_size: float,
        translate_output: bool = True,
        write_json: bool = True,
        run_validation: bool = True,
        agent_parameters: Dict[str, Dict[str, Any]] = None
) -> Dict[str, str]:
    species_names = []
    float_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    with open(input_fp, 'r') as f:
        output = [l.strip() for l in f.readlines()]
        for line in output:
            datum = line.split(' ')[0]
            # Check if the datum is not a float string
            if not float_pattern.match(datum):
                species_names.append(datum)
        f.close()
    species_names = list(set(species_names))

    simularium = SimulariumSmoldynStep(config={
        'output_dest': dest_dir,
        'box_size': box_size,
        'translate_output': translate_output,
        'file_save_name': None,
        'write_json': write_json,
        'run_validation': run_validation,
        'agent_display_parameters': agent_parameters,
    })

    return simularium.update(inputs={
        'results_file': input_fp,
        'species_names': species_names
    })


def make_fallback_serializer_function(process_registry) -> Callable:
    """Creates a fallback function that is called by orjson on data of
    types that are not natively supported. Define and register instances of
    :py:class:`vivarium.core.registry.Serializer()` with serialization
    routines for the types in question."""

    def default(obj: Any) -> Any:
        # Try to lookup by exclusive type
        serializer = process_registry.access(str(type(obj)))
        if not serializer:
            compatible_serializers = []
            for serializer_name in process_registry.list():
                test_serializer = process_registry.access(serializer_name)
                # Subclasses with registered serializers will be caught here
                if isinstance(obj, test_serializer.python_type):
                    compatible_serializers.append(test_serializer)
            if len(compatible_serializers) > 1:
                raise TypeError(
                    f'Multiple serializers ({compatible_serializers}) found '
                    f'for {obj} of type {type(obj)}')
            if not compatible_serializers:
                raise TypeError(
                    f'No serializer found for {obj} of type {type(obj)}')
            serializer = compatible_serializers[0]
            if not isinstance(obj, Process):
                # We don't warn for processes because since their types
                # based on their subclasses, it's not possible to avoid
                # searching through the serializers.
                warn(
                    f'Searched through serializers to find {serializer} '
                    f'for data of type {type(obj)}. This is '
                    f'inefficient.')
        return serializer.serialize(obj)
    return default

