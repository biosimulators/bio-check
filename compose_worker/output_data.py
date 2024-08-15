from tempfile import mkdtemp
from typing import *
from importlib import import_module

import libsbml
try:
    from amici import SbmlImporter, import_model_module, Model, runAmiciSimulation
except ImportError as e:
    print(e)
try:
    from basico import *
except ImportError as e:
    print(e)
try:
    import tellurium as te
except ImportError as e:
    print(e)
try:
    from smoldyn import Simulation
except ImportError as e:
    print(e)
from kisao import AlgorithmSubstitutionPolicy
from biosimulators_utils.config import Config
# from biosimulators_simularium import execute as execute_simularium

from data_model import BiosimulationsRunOutputData
from compatible import COMPATIBLE_UTC_SIMULATORS
from io_worker import read_report_outputs, normalize_smoldyn_output_path_in_root, make_dir


# def generate_smoldyn_simularium(smoldyn_configuration_file: str, output_dest_dir: str, use_json: bool = True, agent_params=None, box_size=None):
#     # 1. make temp dir with config file written to it and set that as archive root
#     temp_archive_root = mkdtemp()
#     with open(smoldyn_configuration_file, 'r') as fh:
#         smoldyn_config = fh.read()
#
#     with open(os.path.join(temp_archive_root, smoldyn_configuration_file.split('/')[-1]), 'w') as f:
#         f.write(smoldyn_config)
#
#     return execute_simularium(working_dir=temp_archive_root, use_json=use_json, output_dir=output_dest_dir, agent_params=agent_params, box_size=box_size)


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

    return output_data


def get_sbml_species_mapping(sbml_fp: str):
    # read file
    sbml_reader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBML(sbml_fp)
    sbml_model_object = sbml_doc.getModel()

    # parse and handle names/ids
    sbml_species_ids = [spec for spec in sbml_model_object.getListOfSpecies()]
    names = list(map(lambda s: s.name, sbml_species_ids))
    vals = [spec.getId() for spec in sbml_species_ids]
    keys = vals if '' in names else names

    return dict(zip(keys, vals))


def run_sbml_tellurium(sbml_fp: str, start, dur, steps):
    simulator = te.loadSBMLModel(sbml_fp)
    floating_species_list = simulator.getFloatingSpeciesIds()
    sbml_species_mapping = get_sbml_species_mapping(sbml_fp)
    output_keys = [list(sbml_species_mapping.keys())[i] for i, spec_id in enumerate(floating_species_list)]

    # in the case that the start time is set to a point after the simulation begins
    if start > 0:
        simulator.simulate(0, start)

    result = simulator.simulate(start, dur, steps + 1)
    outputs = {}
    for index, row in enumerate(result.transpose()):
        for i, name in enumerate(floating_species_list):
            outputs[output_keys[i]] = row

    return outputs


def run_sbml_copasi(sbml_fp: str, start, dur, steps):
    simulator = load_model(sbml_fp)
    sbml_species_mapping = get_sbml_species_mapping(sbml_fp)
    basico_species_ids = list(sbml_species_mapping.keys())

    floating_species_list = list(sbml_species_mapping.values())

    t = np.linspace(start, dur, steps + 1)
    reported_outs = [k for k in sbml_species_mapping.keys()]

    # if start > 0:
        # run_time_course(start_time=0, duration=start, model=simulator, update_model=True)

    _tc = run_time_course_with_output(start_time=t[0], duration=t[-1], values=t, model=simulator, update_model=True, output_selection=reported_outs, use_numbers=True)
    tc = _tc.to_dict()
    output_keys = [list(sbml_species_mapping.keys())[i] for i, spec_id in enumerate(floating_species_list)]

    results = {}
    for i, name in enumerate(floating_species_list):
        results[output_keys[i]] = list(tc.get(basico_species_ids[i]).values())

    return results


def run_sbml_amici(sbml_fp: str, start, dur, steps):
    sbml_reader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBML(sbml_fp)
    sbml_model_object = sbml_doc.getModel()

    sbml_importer = SbmlImporter(sbml_fp)

    model_id = "BIOMD0000000012_url"
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
    sbml_species_ids = [spec for spec in sbml_model_object.getListOfSpecies()]

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
    output_keys = [list(sbml_species_mapping.keys())[i] for i, spec_id in enumerate(floating_species_list)]

    results = {}
    floating_results = dict(zip(
        output_keys,
        list(map(lambda x: result_data.by_id(x), floating_species_list))
    ))
    results = floating_results

    return results


def generate_biosimulator_utc_outputs(omex_fp: str, output_root_dir: str, simulators: List[str] = None, alg_policy="same_framework") -> Dict:
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

        module = import_module(name=f'biosimulators_{sim}.core')
        exec_func = getattr(module, 'exec_sedml_docs_in_combine_archive')
        sim_output_dir = os.path.join(output_root_dir, f'{sim}_outputs')
        if not os.path.exists(sim_output_dir):
            os.mkdir(sim_output_dir)

        # execute simulator-specific simulation
        exec_func(archive_filename=omex_fp, out_dir=sim_output_dir, config=sim_config)

        report_path = os.path.join(sim_output_dir, 'reports.h5')
        sim_data = read_report_outputs(report_path)
        data = sim_data.to_dict() if isinstance(sim_data, BiosimulationsRunOutputData) else {}
        output_data[sim] = data

    return output_data


# TODO: add Vcell and pysces here
SBML_EXECUTORS = dict(zip(
    [data[0] for data in COMPATIBLE_UTC_SIMULATORS],
    [run_sbml_amici, run_sbml_copasi, run_sbml_tellurium]
))


def generate_sbml_utc_outputs(sbml_fp: str, start: int, dur: int, steps: int, simulators: list[str] = None, truth: str = None) -> dict:
    """

    Args:
       sbml_fp: sbml filepath
       start: output start time
       dur: end (output end time)
       steps: number of points
       simulators: list of simulators to generate output from. Defaults to `None`.
       truth: path to the "ground truth" report file. Defaults to `None`.

    """
    output = {}

    # TODO: add VCELL and pysces here
    simulators = simulators or ['amici', 'copasi', 'tellurium']
    for simulator in simulators:
        simulator = simulator.lower()
        result = {}
        simulation_executor = SBML_EXECUTORS[simulator]
        result = simulation_executor(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)

        for species_name in result.keys():
            output_vals = result[species_name]
            if isinstance(output_vals, np.ndarray):
                result[species_name] = output_vals.tolist()

        output[simulator] = result

    if truth is not None:
        output['ground_truth'] = {}
        report_results = read_report_outputs(truth)
        report_data = report_results.to_dict()['data'] if isinstance(report_results, BiosimulationsRunOutputData) else {}

        for datum in report_data:
            spec_name = datum['dataset_label']
            if not spec_name.lower() == 'time':
                spec_data = datum['data']
                output['ground_truth'][spec_name] = spec_data

    return output


def sbml_output_stack(spec_name: str, output):
    stack = []
    for simulator_name, simulator_output in output.items():
        spec_data = simulator_output[spec_name]
        stack.append(spec_data)
    return stack


def _get_output_stack(outputs: dict, spec_id: str):
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
