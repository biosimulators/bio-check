import os
import logging
from tempfile import mkdtemp
from typing import *
from importlib import import_module

import tellurium as te
import libsbml
import numpy as np
from amici import SbmlImporter, import_model_module, Model, runAmiciSimulation
from basico import *
from kisao import AlgorithmSubstitutionPolicy
from biosimulators_utils.config import Config
from biosimulators_simularium import execute as execute_simularium

from data_model import BiosimulationsRunOutputData
# from biosimulator_processes.data_model.service_data_model import BiosimulationsRunOutputData
# from biosimulator_processes.io import standardize_report_outputs
from io_worker import read_report_outputs
from shared import make_dir


def generate_smoldyn_simularium(smoldyn_configuration_file: str, output_dest_dir: str, use_json: bool = True, agent_params=None, box_size=None):
    # 1. make temp dir with config file written to it and set that as archive root
    temp_archive_root = mkdtemp()
    with open(smoldyn_configuration_file, 'r') as fh:
        smoldyn_config = fh.read()

    with open(os.path.join(temp_archive_root, smoldyn_configuration_file.split('/')[-1]), 'w') as f:
        f.write(smoldyn_config)

    return execute_simularium(working_dir=temp_archive_root, use_json=use_json, output_dir=output_dest_dir, agent_params=agent_params, box_size=box_size)


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
        results[output_keys[i]] = np.array(list(tc.get(basico_species_ids[i]).values()))

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


def generate_sbml_utc_outputs(sbml_fp: str, start: int, dur: int, steps: int, truth: str = None) -> dict:
    """

    Args:
       sbml_fp: sbml filepath
       start: output start time
       dur: end (output end time)
       steps: number of points
       truth: path to the "ground truth" report file. Defaults to `None`.

    """
    # amici_results = run_sbml_amici(**params)
    copasi_results = run_sbml_copasi(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)
    tellurium_results = run_sbml_tellurium(sbml_fp=sbml_fp, start=start, dur=dur, steps=steps)
    output = {'copasi': copasi_results, 'tellurium': tellurium_results}  # 'amici': amici_results}

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


def generate_species_output(omex_fp: str, output_root_dir: str, species_name: str, simulators: list[str] = None) -> np.ndarray:
    outputs = generate_biosimulator_outputs(omex_fp, output_root_dir, simulators=simulators)
    return _get_output_stack(outputs, species_name), outputs


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


# def get_species_utc_data(
#         omex_fp: str,
#         species_name: str,
#         output_root_dir: str,
#         ground_truth_source_fp: str = None,
#         simulators: list[str] = None,
#         ) -> pd.DataFrame:
#     spec_index = 0
#     simulator_outputs = generate_biosimulator_outputs(omex_fp, output_root_dir, simulators)
#     for i, spec_name in enumerate(list(simulator_outputs['amici'].keys())):
#         if species_name.lower() in spec_name.lower():
#             spec_index += i
#
#     outs = [
#         simulator_outputs['amici']['data'][spec_index]['data'],
#         simulator_outputs['copasi']['data'][spec_index]['data'],
#         simulator_outputs['tellurium']['data'][spec_index]['data']
#     ]
#     simulator_names = list(simulator_outputs.keys())
#
#     if ground_truth_source_fp:
#         simulator_names.append('ground_truth')
#         ground_truth_results = standardize_report_outputs(ground_truth_source_fp)
#         ground_truth = ground_truth_results['floating_species'][species_name]
#         outs.append(ground_truth)
#
#     return pd.DataFrame(data=np.array(outs), columns=simulator_names)



