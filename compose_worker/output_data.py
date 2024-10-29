from tempfile import mkdtemp
from typing import *
from importlib import import_module

import libsbml
import numpy as np

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

from shared import handle_exception
from data_model import BiosimulationsRunOutputData
from compatible import COMPATIBLE_UTC_SIMULATORS
from io_worker import read_report_outputs, normalize_smoldyn_output_path_in_root, make_dir, read_h5_reports


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
        output_data = {'error': error}

    return output_data


def get_sbml_species_mapping(sbml_fp: str):
    """

    Args:
        - sbml_fp: `str`: path to the SBML model file.

    Returns:
        Dictionary mapping of {sbml_species_names(usually the actual observable name): sbml_species_ids(ids used in the solver)}
    """
    # read file
    sbml_reader = libsbml.SBMLReader()
    sbml_doc = sbml_reader.readSBML(sbml_fp)
    sbml_model_object = sbml_doc.getModel()
    # parse and handle names/ids
    sbml_species_ids = []
    for spec in sbml_model_object.getListOfSpecies():
        spec_name = spec.name
        if not spec_name:
            spec.name = spec.getId()
        if not spec.name == "":
            sbml_species_ids.append(spec)
    names = list(map(lambda s: s.name, sbml_species_ids))
    species_ids = [spec.getId() for spec in sbml_species_ids]
    return dict(zip(names, species_ids))


# 1. add try/except to each sbml output generator: {error: msg}
# 2. in output_stack: return {simname: output}
# 3. In species comparison: for sim in stack.keys(): if isinstance(stack[sim], str): sims.remove(sim), stack.remove(..)


def handle_sbml_exception() -> str:
    import traceback
    from pprint import pformat
    tb_str = traceback.format_exc()
    error_message = pformat(f"SBML Simulation Error:\n{tb_str}")
    
    return error_message


def run_sbml_pysces(sbml_fp: str, start, dur, steps):
    import pysces
    import os
    # # model compilation
    compilation_dir = '/Pysces/psc'  # mkdtemp()
    sbml_filename = sbml_fp.split('/')[-1]
    psc_filename = sbml_filename + '.psc'
    psc_fp = os.path.join(compilation_dir, psc_filename)
    modelname = sbml_filename.replace('.xml', '')
    try:
        # convert sbml to psc
        pysces.model_dir = compilation_dir
        pysces.interface.convertSBML2PSC(sbmlfile=sbml_fp, pscfile=psc_fp)  # sbmldir=os.path.dirname(sbml_fp)

        # instantiate model from compilation contents
        with open(psc_fp, 'r', encoding='utf-8', errors='replace') as F:
            pscS = F.read()

        model = pysces.model(psc_fp, loader='string', fString=pscS)

        # run the simulation with specified time params
        t = np.linspace(start, dur, steps + 1)
        model.sim_time = t
        model.Simulate(1)  # specify userinit=1 to directly use model.sim_time (t) rather than the default

        # get output with mapping of internal species ids to external (shared) species names
        sbml_species_mapping = get_sbml_species_mapping(sbml_fp)
        obs_names = list(sbml_species_mapping.keys())
        obs_ids = list(sbml_species_mapping.values())

        # get raw output data and transpose for correct shape
        # data = model.data_sim.getSpecies().transpose().tolist()
        # remove time reporting TODO: do this more gracefully
        # data.pop(0)
        # return dict(zip(obs_names, data))

        return {
            obs_names[i]: model.data_sim.getSimData(obs_id)[:, 1].tolist()
            for i, obs_id in enumerate(obs_ids)
        }
    except:
        error_message = handle_sbml_exception()
        return {"error": error_message}


def run_sbml_tellurium(sbml_fp: str, start, dur, steps):
    simulator = te.loadSBMLModel(sbml_fp)
    mapping = get_sbml_species_mapping(sbml_fp)

    try:
        # in the case that the start time is set to a point after the simulation begins
        if start > 0:
            simulator.simulate(0, start)

        # run for the specified time
        result = simulator.simulate(start, dur, steps + 1)
        outputs = {}
        for colname in result.colnames:
            if 'time' not in colname:
                for spec_name, spec_id in mapping.items():
                    if colname.replace("[", "").replace("]", "") == spec_id:
                        data = result[colname]
                        outputs[spec_name] = data.tolist()
        return outputs
    except:
        error_message = handle_sbml_exception()
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
        return {spec: tc[spec].values.tolist() for spec in specs}
    except:
        error_message = handle_sbml_exception()
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
        return {"error": error_message}


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


# TODO: add Vcell and pysces here
SBML_EXECUTORS = dict(zip(
    [data[0] for data in COMPATIBLE_UTC_SIMULATORS],
    [run_sbml_amici, run_sbml_copasi, run_sbml_pysces, run_sbml_tellurium]
))


def generate_sbml_utc_outputs(sbml_fp: str, start: int, dur: int, steps: int, simulators: list[str] = None, truth: str = None) -> dict:
    # output = {}
    # # TODO: add VCELL and pysces here
    # sbml_species_ids = list(get_sbml_species_mapping(sbml_fp).values())
    # simulators = simulators or ['amici', 'copasi', 'tellurium']
    # all_output_ids = []
    # for simulator in simulators:
    #     simulator = simulator.lower()
    #     results = {}
    #     simulation_executor = SBML_EXECUTORS[simulator]
    #     sim_result = simulation_executor(sbml_fp, start, dur, steps)
    #     all_output_ids.append(list(sim_result.keys()))
    #     for species_id in sbml_species_ids:
    #         if species_id in sim_result.keys():
    #             output_vals = sim_result[species_id]
    #             if isinstance(output_vals, np.ndarray):
    #                 output_vals = output_vals.tolist()
    #             results[species_id] = output_vals
    #     output[simulator] = results
    # # get the commonly shared output ids
    # shared_output_ids = min(all_output_ids)
    # for simulator_name in output.keys():
    #     sim_data = {}
    #     for spec_id in output[simulator_name].keys():
    #         if spec_id in shared_output_ids:
    #             sim_data[spec_id] = output[simulator_name][spec_id]
    #     output[simulator_name] = sim_data
    output = {}
    sbml_species_ids = list(get_sbml_species_mapping(sbml_fp).keys())
    simulators = simulators or ['amici', 'copasi', 'tellurium', 'pysces']
    all_output_ids = []
    for simulator in simulators:
        results = {}
        simulator = simulator.lower()
        simulation_executor = SBML_EXECUTORS[simulator]
        sim_result = simulation_executor(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)
        # sim_result = None
        # if simulator == 'amici':
        #     sim_result = run_sbml_amici(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)
        # elif simulator == 'copasi':
        #     sim_result = run_sbml_copasi(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)
        # elif simulator == 'pysces':
        #     sim_result = run_sbml_pysces(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)
        # elif simulator == 'tellurium':
        #     sim_result = run_sbml_tellurium(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)

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


# def sbml_output_stack(spec_name: str, output):
#     stack = []
#     for simulator_name in output.keys():
#         spec_data = output[simulator_name].get(spec_name)
#         if spec_data is not None:
#             stack.append(spec_data)
#     return stack


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


def _get_output_stack(outputs: dict, spec_id: str) -> np.ndarray:
    output_stack = []
    for sim_name in outputs.keys():
        sim_data = outputs[sim_name]
        if isinstance(sim_data, dict):
            for spec_name in sim_data.keys():
                if spec_name == spec_id:
                    output_stack.append(sim_data[spec_name])

    return np.stack(output_stack)


def __get_output_stack(outputs: dict, spec_id: str):
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


