import logging
import os
from typing import *
from importlib import import_module

import numpy as np
from kisao import AlgorithmSubstitutionPolicy
from biosimulators_utils.config import Config

from worker.service.io_worker import get_sbml_species_mapping
from worker.service.data_generator import SBML_EXECUTORS
from worker.service.log_config import setup_logging
from worker.service.data_model import BiosimulationsRunOutputData
from worker.service.io_worker import read_report_outputs, make_dir, read_h5_reports


# logging TODO: implement this.
logger = logging.getLogger("biochecknet.worker.verification.log")
setup_logging(logger)


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


