import logging
import math
import os
import tempfile
from typing import *
from abc import ABC, abstractmethod
from process_bigraph import Composite

import numpy as np
import pandas as pd

from log_config import setup_logging
from shared import unique_id, BUCKET_NAME, PROCESS_TYPES
from io_worker import get_sbml_species_names, get_sbml_model_file_from_archive, read_report_outputs, download_file, format_smoldyn_configuration, write_uploaded_file
from output_data import (
    generate_biosimulator_utc_outputs,
    _get_output_stack,
    sbml_output_stack,
    generate_sbml_utc_outputs,
    get_sbml_species_mapping,
    run_smoldyn
)
from bigraph_steps import generate_simularium_file


# -- WORKER: "Toolkit" => Has all of the tooling necessary to process jobs.


# logging
logger = logging.getLogger(__name__)
setup_logging(logger)


class Worker(ABC):
    job_params: Dict
    job_id: str
    job_result: Dict | None

    def __init__(self, job: Dict):
        """
        Args:
            job: job parameters received from the supervisor (who gets it from the db) which is a document from the pending_jobs collection within mongo.
        """
        self.job_params = job
        self.job_id = self.job_params['job_id']
        self.job_result = {}

        # for parallel processing in a pool of workers. TODO: eventually implement this.
        self.worker_id = unique_id()

    @abstractmethod
    async def run(self):
        pass


class SimulationRunWorker(Worker):
    def __init__(self, job: Dict):
        super().__init__(job=job)

    async def run(self):
        # check which endpoint methodology to implement
        out_dir = tempfile.mkdtemp()
        source_fp = self.job_params['path']
        local_fp = download_file(source_blob_path=source_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

        # case: is a smoldyn job
        if local_fp.endswith('.txt'):
            await self.run_smoldyn(local_fp)
        # case: is utc job
        elif local_fp.endswith('.xml'):
            await self.run_utc(local_fp)

        return self.job_result

    async def run_smoldyn(self, local_fp: str):
        # format model file for disabling graphics
        format_smoldyn_configuration(filename=local_fp)

        # get job params
        duration = self.job_params.get('duration')
        dt = self.job_params.get('dt')
        initial_species_state = self.job_params.get('initial_molecule_state')  # not yet implemented

        # execute simularium, pointing to a filepath that is returned by the run smoldyn call
        result = run_smoldyn(model_fp=local_fp, duration=duration, dt=dt)

        # TODO: Instead use the composition framework to do this

        # write the aforementioned output file (which is itself locally written to the temp out_dir, to the bucket if applicable
        results_file = result.get('results_file')
        if results_file is not None:
            uploaded_file_location = await write_uploaded_file(job_id=self.job_id, uploaded_file=results_file, bucket_name=BUCKET_NAME, extension='.txt')
            self.job_result = {'results_file': uploaded_file_location}
        else:
            self.job_result = result

    async def run_utc(self, local_fp: str):
        start = self.job_params['start']
        end = self.job_params['end']
        steps = self.job_params['steps']
        simulator = self.job_params['simulator']

        # TODO: instead use the composition framework to do this!

        result = generate_sbml_utc_outputs(sbml_fp=local_fp, start=start, dur=end, steps=steps, simulators=[simulator])
        self.job_result = result[simulator]


class VerificationWorker(Worker):
    def __init__(self, job: Dict):
        super().__init__(job=job)

    async def run(self, selection_list: List[str] = None) -> Dict:
        # process simulation
        input_fp = self.job_params['path']
        selection_list = self.job_params.get('selection_list')
        if input_fp.endswith('.omex'):
            self._execute_omex_job()
        elif input_fp.endswith('.xml'):
            self._execute_sbml_job()

        # select data if applicable
        selections = self.job_params.get("selection_list", selection_list)
        if selections is not None:
            self.job_result = self._select_observables(job_result=self.job_result, observables=selections)

        # calculate rmse
        try:
            self.job_result['rmse'] = self._calculate_pairwise_rmse()
        except Exception as e:
            self.job_result['rmse'] = {'error': str(e)}

        # simulators = self.job_params.get('simulators')
        # # include expected results in rmse if applicable
        # if self.job_params.get('expected_results') is not None:
        #     simulators.append('expected_results')
        # # calc rmse for each simulator
        # for simulator in simulators:
        #     try:
        #         self.job_result['rmse'][simulator] = self._calculate_inter_simulator_rmse(target_simulator=simulator)
        #     except:
        #         self.job_result['rmse'][simulator] = {}

        return self.job_result

    def _calculate_inter_simulator_rmse(self, target_simulator):
        # extract data fields
        spec_data = self.job_result

        # iterate through observables
        mse_values = []
        for observable, sim_details in spec_data.items():
            mse_data = sim_details['mse'][target_simulator]

            # exclude self-comparison and collect MSE values with other simulators
            for sim, mse in mse_data.items():
                if sim != target_simulator:
                    mse_values.append(mse)

        # calculate the mean of the collected MSE values
        if mse_values:
            mean_mse = sum(mse_values) / len(mse_values)

            # return the square root of the mean MSE (RMSE)
            return math.sqrt(mean_mse)
        else:
            # handle case where no MSE values are present (to avoid division by zero)
            return 0.0

    def _calculate_pairwise_rmse(self) -> dict:
        # get input data
        spec_data = self.job_result
        simulators = self.job_params['simulators']
        if self.job_params.get('expected_results') is not None:
            simulators.append('expected_results')
        n = len(simulators)

        # set up empty matrix
        rmse_matrix = np.zeros((n, n))

        # enumerate over i,j of simulators in a matrix
        for i, sim_i in enumerate(simulators):
            for j, sim_j in enumerate(simulators):
                if i != j:
                    mse_values = []
                    for observable, observable_data in spec_data.items():
                        if not isinstance(observable_data, str):
                            mse_data = observable_data['mse']
                            if sim_j in mse_data:
                                # mse_data[sim_j] is a dict containing MSEs with other simulators
                                for comparison_sim, mse_value in mse_data[sim_j].items():
                                    if comparison_sim == sim_i:
                                        mse_values.append(mse_value)
                    if mse_values:
                        mean_mse = sum(mse_values) / len(mse_values)
                        rmse_matrix[i, j] = math.sqrt(mean_mse)
                    else:
                        # TODO: make this more robust
                        rmse_matrix[i, j] = np.nan
                else:
                    rmse_matrix[i, j] = 0.0

        return pd.DataFrame(rmse_matrix, columns=simulators, index=simulators).to_dict()

    def _select_observables(self, job_result, observables: List[str] = None) -> Dict:
        """Select data from the input data that is passed which should be formatted such that the data has mappings of observable names
            to dicts in which the keys are the simulator names and the values are arrays. The data must have content accessible at: `data['content']['results']`.
        """
        outputs = job_result.copy()
        result = {}
        data = job_result

        # case: results from sbml
        if isinstance(data, dict):
            for name, obs_data in data.items():
                if name in observables:
                    result[name] = obs_data
            outputs = result
        # case: results from omex
        elif isinstance(data, list):
            for i, datum in enumerate(data):
                name = datum['species_name']
                if name not in observables:
                    print(f'Name: {name} not in observables')
                    data.pop(i)
            outputs = data

        return outputs

    def _execute_sbml_job(self):
        params = None
        out_dir = tempfile.mkdtemp()
        source_fp = self.job_params['path']
        source_report_fp = self.job_params.get('expected_results')

        # download sbml file
        local_fp = download_file(source_blob_path=source_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

        # get ground truth from bucket if applicable
        local_report_fp = None
        if source_report_fp is not None:
            local_report_fp = download_file(source_blob_path=source_report_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

        simulators = self.job_params.get('simulators', ['amici', 'copasi', 'tellurium'])
        include_outs = self.job_params.get('include_outputs', False)
        comparison_id = self.job_params.get('job_id')
        output_start = self.job_params.get('start')
        end = self.job_params.get('end', 10)
        steps = self.job_params.get('steps', 100)
        rtol = self.job_params.get('rTol')
        atol = self.job_params.get('aTol')

        try:
            result = self._run_comparison_from_sbml(
                sbml_fp=local_fp,
                start=output_start,
                dur=end,
                steps=steps,
                rTol=rtol,
                aTol=atol,
                ground_truth=local_report_fp,
                simulators=simulators
            )
            self.job_result = result
        except Exception as e:
            self.job_result = {"bio-composer-message": f"Job for {self.job_params['comparison_id']} could not be completed because:\n{str(e)}"}
            raise e

    def _execute_omex_job(self):
        params = None
        out_dir = tempfile.mkdtemp()
        source_fp = self.job_params['path']
        source_report_fp = self.job_params.get('expected_results')

        # download sbml file
        local_fp = download_file(source_blob_path=source_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)

        # get ground truth from bucket if applicable
        truth_vals = None
        local_report_fp = None
        if source_report_fp is not None:
            local_report_fp = download_file(source_blob_path=source_report_fp, out_dir=out_dir, bucket_name=BUCKET_NAME)
            truth_vals = read_report_outputs(local_report_fp)

        try:
            simulators = self.job_params.get('simulators', [])
            include_outs = self.job_params.get('include_outputs', False)
            tol = self.job_params.get('rTol')
            atol = self.job_params.get('aTol')
            comparison_id = self.job_params.get('job_id')

            result = self._run_comparison_from_omex(
                path=local_fp,
                simulators=simulators,
                out_dir=out_dir,
                include_outputs=include_outs,
                comparison_id=comparison_id,
                truth_vals=truth_vals
            )
            self.job_result = result
        except Exception as e:
            self.job_result = {"bio-composer-message": f"Job for {self.job_params['job_id']} could not be completed because:\n{str(e)}"}
            raise e

    def _run_comparison_from_omex(
            self,
            path: str,
            simulators: List[str],
            out_dir: str,
            include_outputs: bool = True,
            comparison_id: str = None,
            truth_vals=None,
            rTol=None,
            aTol=None
    ) -> Dict:
        """Execute a Uniform Time Course comparison for ODE-based simulators from Biosimulators."""
        # download the omex file from GCS
        # source_blob_name = path.replace('gs://bio-check-requests-1', '')  # Assuming omex_fp is the blob name in GCS
        # local_omex_fp = os.path.join(out_dir, path.split('/')[-1])
        # download_blob(bucket_name=BUCKET_NAME, source_blob_name=path, destination_file_name=local_omex_fp)

        # download the report file from GCS if applicable
        # if ground_truth_report_path is not None:
        #     source_report_blob_name = ground_truth_report_path.replace('gs://bio-check-requests-1', '')
        #     # local_report_path = os.path.join(out_dir, source_report_blob_name.split('/')[-1])
        #     local_report_path = os.path.join(out_dir, ground_truth_report_path.split('/')[-1])
        #     truth_vals = read_report_outputs(ground_truth_report_path)
        # else:
        #     truth_vals = None

        # run comparison
        comparison_id = comparison_id or 'biosimulators-utc-comparison'
        ground_truth_data = truth_vals.to_dict() if not isinstance(truth_vals, type(None)) else truth_vals

        comparison = self._generate_omex_utc_comparison(
            omex_fp=path,  # path,
            out_dir=out_dir,  # TODO: replace this with an s3 endpoint.
            simulators=simulators,
            comparison_id=comparison_id,
            ground_truth=ground_truth_data,
            rTol=rTol,
            aTol=aTol
        )
        return comparison
        # parse data for return vals
        # spec_comparisons = []
        # for spec_name, comparison_data in comparison['results'].items():
        #     species_comparison = UtcSpeciesComparison(
        #         mse=comparison_data['mse'],
        #         proximity=comparison_data['proximity'],
        #         output_data=comparison_data.get('output_data') if include_outputs else {},
        #         species_name=spec_name)
        #     spec_comparisons.append(species_comparison)
#
        # return UtcComparison(
        #     results=spec_comparisons,
        #     id=comparison_id,
        #     simulators=simulators)

    def _run_comparison_from_sbml(self, sbml_fp, start, dur, steps, rTol=None, aTol=None, simulators=None, ground_truth=None) -> Dict:
        species_mapping = get_sbml_species_mapping(sbml_fp)
        results = {}
        output_data = self._generate_formatted_sbml_outputs(sbml_filepath=sbml_fp, start=start, dur=dur, steps=steps, simulators=simulators)
        sims = simulators or list(output_data.keys())
        for simulator in sims:
            sim_data = output_data[simulator]
            # for i, species_name in enumerate(list(species_mapping.keys())):
            for i, species_name in enumerate(list(sim_data.keys())):
                species_comparison = self._generate_sbml_utc_species_comparison(
                    output_data=output_data,
                    species_name=species_name,
                    rTol=rTol,
                    aTol=aTol,
                    ground_truth=ground_truth
                )
                results[list(species_mapping.keys())[i]] = species_comparison

        return results

    def _generate_omex_utc_comparison(self, omex_fp, out_dir, simulators, comparison_id, ground_truth=None, rTol=None, aTol=None):
        results = {}

        # generate the data
        output_data = generate_biosimulator_utc_outputs(omex_fp=omex_fp, output_root_dir=out_dir, simulators=simulators, alg_policy="same_framework")

        # generate the species comparisons
        # for i, species in enumerate(sbml_species_names):
        observable_names = []
        for simulator_name in output_data.keys():
            for observable_name in output_data[simulator_name].keys():
                observable_names.append(observable_name)
        names = list(set(observable_names))

        for species in names:
            # TODO: reimplement this!
            ground_truth_data = None
            # if ground_truth:
            #     for data in ground_truth['data']:
            #         if data['dataset_label'] == species:
            #             ground_truth_data = data['data']

            # generate species comparison
            results[species] = self._generate_omex_utc_species_comparison(
                output_data=output_data,
                species_name=species,
                simulators=simulators,
                ground_truth=ground_truth_data,
                rTol=rTol,
                aTol=aTol
            )

        return results

    def _generate_omex_utc_species_comparison(self, output_data, species_name, simulators, ground_truth=None, rTol=None, aTol=None):
        # output_data = generate_biosimulator_utc_outputs(omex_fp=omex_fp, output_root_dir=out_dir, simulators=simulators, alg_policy="same_framework")
        outputs = _get_output_stack(output_data, species_name)
        methods = ['mse', 'proximity']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth, rtol=rTol, atol=aTol).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        for simulator_name in output_data.keys():
            sim_output = output_data[simulator_name]
            for spec_name, spec_output in sim_output.items():
                if spec_name == species_name:
                    data = output_data[simulator_name][spec_name]
                    results['output_data'][simulator_name] = data.tolist() if isinstance(data, np.ndarray) else data

        return results

    def __generate_omex_utc_species_comparison(self, omex_fp, out_dir, species_name, simulators, ground_truth=None, rTol=None, aTol=None):
        output_data = generate_biosimulator_utc_outputs(omex_fp=omex_fp, output_root_dir=out_dir, simulators=simulators, alg_policy="same_framework")
        outputs = _get_output_stack(output_data, species_name)
        methods = ['mse', 'proximity']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth, rtol=rTol, atol=aTol).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        for simulator_name in output_data.keys():
            for output in output_data[simulator_name]['data']:
                if output['dataset_label'] in species_name:
                    data = output['data']
                    results['output_data'][simulator_name] = data.tolist() if isinstance(data, np.ndarray) else data
        return results

    def _generate_formatted_sbml_outputs(self, sbml_filepath, start, dur, steps, simulators, ground_truth=None):
        return generate_sbml_utc_outputs(sbml_fp=sbml_filepath, start=start, dur=dur, steps=steps, simulators=simulators, truth=ground_truth)

    def _generate_sbml_utc_species_comparison(self, output_data, species_name, ground_truth=None, rTol=None, aTol=None):
        outputs = sbml_output_stack(spec_name=species_name, output=output_data)
        simulators = list(output_data.keys())
        methods = ['mse', 'proximity']
        matrix_vals = list(map(
            lambda m: self._generate_species_comparison_matrix(outputs=outputs, simulators=simulators, method=m, ground_truth=ground_truth, rtol=rTol, atol=aTol).to_dict(),
            methods
        ))
        results = dict(zip(methods, matrix_vals))
        results['output_data'] = {}
        for simulator_name in output_data.keys():
            for spec_name, output in output_data[simulator_name].items():
                if species_name in spec_name:
                    data = output_data[simulator_name][spec_name]
                    results['output_data'][simulator_name] = data.tolist() if isinstance(data, np.ndarray) else data
        return results

    def _generate_species_comparison_matrix(
            self,
            outputs: Union[np.ndarray, List[np.ndarray]],
            simulators: List[str],
            method: Union[str, any] = 'proximity',
            rtol: float = None,
            atol: float = None,
            ground_truth: np.ndarray = None
    ) -> pd.DataFrame:
        """Generate a Mean Squared Error comparison matrix of arr1 and arr2, indexed by simulators by default,
            or an AllClose Tolerance routine result if `method` is set to `proximity`.

            Args:
                outputs: list of output arrays.
                simulators: list of simulator names.
                method: pass one of either: `mse` to perform a mean-squared error calculation
                    or `proximity` to perform a pair-wise proximity tolerance test using `np.allclose(outputs[i], outputs[i+1])`.
                rtol:`float`: relative tolerance for comparison if `proximity` is used.
                atol:`float`: absolute tolerance for comparison if `proximity` is used.
                ground_truth: If passed, this value is compared against each simulator in simulators. Currently, this
                    field is agnostic to any verified/validated source, and we trust that the user has verified it. Defaults
                    to `None`.

            Returns:
                Pandas dataframe representing a comparison matrix where `i` and `j` are both indexed by the
                    simulators involved. The aforementioned simulators involved will also include the `ground_truth` value
                    within the indices if one is passed.
        """
        # TODO: implement the ground truth
        _simulators = simulators.copy()
        _outputs = outputs.copy()
        if isinstance(_outputs, np.ndarray):
            _outputs = _outputs.tolist()

        if ground_truth is not None:
            _simulators.append('expected_results')
            _outputs.append(ground_truth)

        use_tol_method = method.lower() == 'proximity'
        matrix_dtype = np.float64 if not use_tol_method else bool
        num_simulators = len(_simulators)
        mse_matrix = np.zeros((num_simulators, num_simulators), dtype=matrix_dtype)

        # fill the matrices with the calculated values
        for i in range(len(_simulators)):
            for j in range(i, len(_simulators)):
                output_i = _outputs[i]
                output_j = _outputs[j]
                method_type = method.lower()
                result = self.calculate_mse(output_i, output_j) if method_type == 'mse' else self.compare_arrays(arr1=output_i, arr2=output_j, rtol=rtol, atol=atol)

                mse_matrix[i, j] = result
                if i != j:
                    mse_matrix[j, i] = mse_matrix[i, j]

        return pd.DataFrame(mse_matrix, index=_simulators, columns=_simulators)

    def calculate_mse(self, a, b) -> float:
        if isinstance(a, list):
            a = np.array(a)
        if isinstance(b, list):
            b = np.array(b)
        return np.mean((a - b) ** 2)

    def compare_arrays(self, arr1: np.ndarray, arr2: np.ndarray, atol=None, rtol=None) -> bool:
        """Original methodology copied from biosimulations runutils."""
        max1 = max(arr1)
        max2 = max(arr2)
        aTol = atol or max(1e-3, max1 * 1e-5, max2 * 1e-5)
        rTol = rtol or 1e-4
        return np.allclose(arr1, arr2, rtol=rTol, atol=aTol)


class CompositionWorker(Worker):
    def __init__(self, job):
        super().__init__(job=job)

    async def run(self):
        # extract params
        duration = self.job_params['duration']
        composite_doc = self.job_params['composite_doc']

        # create emitter for results if not already:
        result_emitter_spec = {
            '_type': 'step',
            "address": "local:ram-emitter",
            "config": {
                "emit": {
                    "time": "float",
                    "floating_species_concentrations": "tree[float]"
                }
            },
            "inputs": {
                "time": ["time_store"],
                "floating_species_concentrations": ["floating_species_concentrations_store"],
            }
        }

        composite_doc['results'] = composite_doc.get('results', result_emitter_spec)

        # instantiate composition
        composition = Composite(config=composite_doc, core=PROCESS_TYPES)

        # run composition and set results
        composition.run(duration)
        self.job_result = composition.gather_results()

        return self.job_result


class FilesWorker(Worker):
    def __init__(self, job):
        super().__init__(job)

    async def run(self):
        job_id = self.job_params['job_id']
        input_path = self.job_params.get('path')

        try:
            # is a job related to a client file upload
            if input_path is not None:
                # download the input file
                dest = tempfile.mkdtemp()
                local_input_path = download_file(source_blob_path=input_path, bucket_name=BUCKET_NAME, out_dir=dest)

                # case: is a smoldyn output file and thus a simularium job
                if local_input_path.endswith('.txt'):
                    await self._run_simularium(job_id=job_id, input_path=local_input_path, dest=dest)
        except Exception as e:
            self.job_result = {'results': str(e)}

        return self.job_result

    async def _run_simularium(self, job_id: str, input_path: str, dest: str):
        # get parameters from job
        box_size = self.job_params['box_size']
        translate = self.job_params['translate_output']
        validate = self.job_params['validate_output']
        params = self.job_params.get('agent_parameters')

        # generate file
        result = generate_simularium_file(
            input_fp=input_path,
            dest_dir=dest,
            box_size=box_size,
            translate_output=translate,
            run_validation=validate,
            agent_parameters=params
        )

        # upload file to bucket
        results_file = result.get('simularium_file')
        uploaded_file_location = None
        if results_file is not None:
            if not results_file.endswith('.simularium'):
                results_file += '.simularium'
            uploaded_file_location = await write_uploaded_file(job_id=job_id, bucket_name=BUCKET_NAME, uploaded_file=results_file, extension='.simularium')

        # set uploaded file as result
        self.job_result = {'results_file': uploaded_file_location}
