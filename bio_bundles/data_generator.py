import logging
import os
import traceback
from tempfile import mkdtemp
import uuid
from pprint import pformat
from typing import *
from abc import abstractmethod
from logging import warn, Logger
from uuid import uuid4

import libsbml
import numpy as np
from process_bigraph import Step, Process
from process_bigraph.composite import Emitter, ProcessTypes, Composite
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from simulariumio import InputFileData, UnitData, DisplayData, DISPLAY_TYPE
from simulariumio.smoldyn import SmoldynData

from bio_bundles.data_model.bigraph import time_course_node_spec
from bio_bundles.compatible import COMPATIBLE_UTC_SIMULATORS
from bio_bundles.io import normalize_smoldyn_output_path_in_root, get_sbml_species_mapping
from bio_bundles.simularium_utils import calculate_agent_radius, translate_data_object, write_simularium_file
from bio_bundles.utils import handle_exception, handle_sbml_exception

# logging TODO: implement this.
logger: Logger = logging.getLogger("compose.worker.data_generator.log")

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


# -- functions related to generating simulation output data (ODE, PDE, stochastic particle, and more) using the process-bigraph engine -- #

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
    """
    Method used specifically to generate and format the results of composition for use with the REST API.
    """
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

def generate_sbml_outputs(sbml_fp: str, start: int, dur: int, steps: int, simulators: list[str] = None, truth: str = None) -> dict:
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

    return final_output


# -- process-bigraph implementations -- #



