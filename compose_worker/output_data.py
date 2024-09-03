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
        if not spec.name == "":
            sbml_species_ids.append(spec)
    names = list(map(lambda s: s.name, sbml_species_ids))
    species_ids = [spec.getId() for spec in sbml_species_ids]
    return dict(zip(names, species_ids))


def sbml_to_psc(sbml_fp: str, compilation_dir: str = None):
    import pysces
    compilation_dir = mkdtemp()
    modelname = sbml_fp.split('/')[-1].replace('.psc', '')
    pysces.interface.convertSBML2PSC(sbml_fp, sbmldir=None, pscfile=modelname, pscdir=compilation_dir)
    return sbml_fp + '.psc'


def load_psc_model(psc_fp: str, modelname=None):
    import pysces
    F = open(psc_fp, 'r', encoding='UTF-8')
    pscS = F.read()
    F.close()
    modelname = psc_fp.split('/')[-1].replace('.psc', '')

    return pysces.model(modelname, loader='string', fString=pscS)


def load_pysces_model(sbml_fp: str, compilation_dir: str = None):
    psc_fp = sbml_to_psc(sbml_fp, compilation_dir)
    return load_psc_model(psc_fp)


def _run_sbml_pysces(sbml_fp: str, start: int, dur: int, steps: int):
    sbml_species_mapping = get_sbml_species_mapping(sbml_fp)
    obs_names = list(sbml_species_mapping.keys())
    obs_ids = list(sbml_species_mapping.values())
    model = load_pysces_model(sbml_fp=sbml_fp)
    model.sim_start = start
    model.sim_stop = dur
    model.sim_points = steps + 1
    model.Simulate()
    return {
        obs_names[i]: model.data_sim.getSimData(obs_id)
        for i, obs_id in enumerate(obs_ids)
    }


def run_sbml_pysces(sbml_fp: str, start, dur, steps):
    import pysces
    # # model compilation
    import os
    compilation_dir = mkdtemp()
    sbml_filename = sbml_fp.split('/')[-1]
    psc_filename = sbml_filename + '.psc'
    psc_fp = os.path.join(compilation_dir, psc_filename)
    modelname = sbml_filename.replace('.xml', '')
    # convert sbml to psc
    pysces.interface.convertSBML2PSC(sbml_fp, sbmldir=os.path.dirname(sbml_fp))

    # instantiate model from compilation contents
    with open(psc_fp, 'rb', encoding='UTF-8') as F:
        pscS = F.read()
        # F.close()
        model = pysces.model(modelname, loader='string', fString=pscS)

    # load the sbml model
    # model = pysces.loadSBML(sbmlfile=sbml_fp, sbmldir=os.path.dirname(sbml_fp), pscfile=psc_fp, pscdir=compilation_dir)

    # run the simulation with specified time params
    model.sim_start = start
    model.sim_stop = dur
    model.sim_points = steps + 1
    model.Simulate()

    # get output with mapping of internal species ids to external (shared) species names
    sbml_species_mapping = get_sbml_species_mapping(sbml_fp)
    obs_names = list(sbml_species_mapping.keys())
    obs_ids = list(sbml_species_mapping.values())

    return {
        obs_names[i]: model.data_sim.getSimData(obs_id)
        for i, obs_id in enumerate(obs_ids)
    }


def run_sbml_tellurium(sbml_fp: str, start, dur, steps):
    simulator = te.loadSBMLModel(sbml_fp)
    floating_species_list = simulator.getFloatingSpeciesIds()
    sbml_species_names = list(get_sbml_species_mapping(sbml_fp).keys())

    # in the case that the start time is set to a point after the simulation begins
    if start > 0:
        simulator.simulate(0, start)

    # run for the specified time
    result = simulator.simulate(start, dur, steps + 1)
    outputs = {}
    for index, row in enumerate(result.transpose()):
        if not index == 0:
            for i, name in enumerate(floating_species_list):
                outputs[sbml_species_names[i]] = row
    return outputs


def run_sbml_copasi(sbml_fp: str, start, dur, steps):
    simulator = load_model(sbml_fp)
    species_info = get_species(model=simulator)
    sbml_ids = list(species_info['sbml_id'])  # matches libsbml and solver ids
    basico_species_names = list(species_info.index)  # sbml species NAMES, as copasi is familiar with the names
    # remove EmptySet reference if applicable
    for _id in basico_species_names:
        if "EmptySet" in _id:
            sbml_ids.remove(_id)
            basico_species_names.remove(_id)
    # set time
    t = np.linspace(start, dur, steps + 1)
    # get outputs
    names = [f'[{name}]' for name in basico_species_names]
    _tc = run_time_course_with_output(output_selection=names, start_time=t[0], duration=t[-1], values=t, model=simulator, update_model=True, use_numbers=True)
    tc = _tc.to_dict()
    results = {}
    for i, name in enumerate(names):
        tc_observable_data = tc.get(name)
        if tc_observable_data is not None:
            results[basico_species_names[i]] = list(tc_observable_data.values())
    return results


def run_sbml_amici(sbml_fp: str, start, dur, steps):
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

    return output_data


def generate_biosimulator_utc_outputs(omex_fp: str, output_root_dir: str, simulators: list[str] = None, alg_policy="same_framework") -> dict:
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

        sim_data = read_h5_reports(report_path)
        data = sim_data.to_dict() if isinstance(sim_data, BiosimulationsRunOutputData) else sim_data
        output_data[sim] = data

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
    simulators = simulators or ['amici', 'copasi', 'pysces', 'tellurium']
    all_output_ids = []
    for simulator in simulators:
        try:
            simulator = simulator.lower()
            results = {}
            simulation_executor = SBML_EXECUTORS[simulator]
            sim_result = simulation_executor(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)
            all_output_ids.append(list(sim_result.keys()))

            for species_id in sbml_species_ids:
                if species_id in sim_result.keys():
                    output_vals = sim_result[species_id]
                    if isinstance(output_vals, np.ndarray):
                        output_vals = output_vals.tolist()
                    results[species_id] = output_vals
            output[simulator] = results
        except Exception as e:
            print(str(e))
            output[simulator] = {}

    # get the commonly shared output ids
    final_output = {}
    shared_output_ids = min(all_output_ids)
    for simulator_name in output.keys():
        sim_data = {}
        for spec_id in output[simulator_name].keys():
            if spec_id in shared_output_ids:
                sim_data[spec_id] = output[simulator_name][spec_id]
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


def sbml_output_stack(spec_name: str, output):
    stack = []
    for simulator_name in output.keys():
        spec_data = output[simulator_name].get(spec_name)
        if spec_data is not None:
            stack.append(spec_data)
    return stack


def _get_output_stack(outputs: dict, spec_id: str) -> np.ndarray:
    output_stack = []
    for sim_name in outputs.keys():
        sim_data = outputs[sim_name]
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


def test():
    from output_data import generate_sbml_utc_outputs
    sbml_fp = "/Users/alexanderpatrie/Downloads/BIOMD0000000005_url.xml"
    r = generate_sbml_utc_outputs(sbml_fp, 0, 100, 1000, ['amici', 'copasi', 'tellurium'])
    for sim in r.keys():
        print(r[sim].keys())
