import os
from typing import *
from importlib import import_module

import pandas as pd
import numpy as np
from process_bigraph import pp

from biosimulator_processes.api.compare import generate_comparison
from biosimulator_processes.io import read_report_outputs, standardize_report_outputs
from biosimulators_amici.core import exec_sedml_docs_in_combine_archive as exec_amici
from biosimulators_copasi.core import exec_sedml_docs_in_combine_archive as exec_copasi
from biosimulators_tellurium.core import exec_sedml_docs_in_combine_archive as exec_tellurium
from biosimulators_utils.report.data_model import ReportFormat
from biosimulators_utils.config import Config


def generate_biosimulator_outputs(omex_fp: str, output_root_dir: str, simulators: list[str] = None) -> dict:
    if not os.path.exists(output_root_dir):
        os.mkdir(output_root_dir)

    output_data = {}
    sims = simulators or ['amici', 'copasi', 'tellurium']
    for sim in sims:
        module = import_module(name=f'biosimulators_{sim}.core')
        exec_func = getattr(module, 'exec_sedml_docs_in_combine_archive')

        sim_output_dir = os.path.join(output_root_dir, f'{sim}_outputs')
        if not os.path.exists(sim_output_dir):
            os.mkdir(sim_output_dir)

        sim_config = Config(LOG=False)
        exec_func(archive_filename=omex_fp, out_dir=sim_output_dir, config=sim_config)
        report_path = os.path.join(sim_output_dir, 'reports.h5')
        sim_data = read_report_outputs(report_path)
        output_data[sim] = sim_data.to_dict()

    return output_data


def generate_species_output(omex_fp: str, output_root_dir: str, species_name: str, simulators: list[str] = None) -> np.ndarray:
    outputs = generate_biosimulator_outputs(omex_fp, output_root_dir, simulators=simulators)
    return _get_output_stack(outputs, species_name)


def _get_output_stack(outputs: dict, spec_id: str):
    output_stack = []
    for sim_name in outputs.keys():
        sim_data = outputs[sim_name]['data']
        for data_index, data in enumerate(sim_data):
            data_id = data['dataset_label']
            if data_id == spec_id:
                print(spec_id, data_id)
                output_stack.append(sim_data[data_index]['data'])
            else:
                pass
    return np.stack(output_stack)


def get_species_utc_data(
        omex_fp: str,
        species_name: str,
        output_root_dir: str,
        ground_truth_source_fp: str = None,
        simulators: list[str] = None,
        ) -> pd.DataFrame:
    spec_index = 0
    simulator_outputs = generate_biosimulator_outputs(omex_fp, output_root_dir, simulators)
    for i, spec_name in enumerate(list(simulator_outputs['amici'].keys())):
        if species_name.lower() in spec_name.lower():
            spec_index += i

    outs = [
        simulator_outputs['amici']['data'][spec_index]['data'],
        simulator_outputs['copasi']['data'][spec_index]['data'],
        simulator_outputs['tellurium']['data'][spec_index]['data']
    ]
    simulator_names = list(simulator_outputs.keys())

    if ground_truth_source_fp:
        simulator_names.append('ground_truth')
        ground_truth_results = standardize_report_outputs(ground_truth_source_fp)
        ground_truth = ground_truth_results['floating_species'][species_name]
        outs.append(ground_truth)

    return pd.DataFrame(data=np.array(outs), columns=simulator_names)



