'''TODO: Implement this in ODESimulation base class:

    @abc.abstractmethod
    def _set_model_changes(self, **changes):
        self._set_floating_species_changes(**changes)
        self._set_global_parameters_changes(**changes)
        self._set_reactions_changes(**changes)


    @abc.abstractmethod
    def _set_floating_species_changes(self):
        """Sim specific method for starting values relative to this property."""
        pass

    @abc.abstractmethod
    def _set_global_parameters_changes(self):
        """Sim specific method for starting values relative to this property."""
        pass

    @abc.abstractmethod
    def _set_reactions_changes(self):
        """Sim specific method for starting values relative to this property."""
        pass
'''


import abc
import logging
import os.path
from tempfile import mkdtemp
from typing import *
from typing import Dict, List, Any

import libsbml
import numpy as np
import tellurium as te
from numpy import ndarray, dtype
from tellurium.roadrunner.extended_roadrunner import ExtendedRoadRunner
from amici import (
    amici,
    sbml_import,
    SbmlImporter,
    import_model_module,
    runAmiciSimulation,
    ReturnDataView)
from COPASI import CDataModel
from pandas import DataFrame
from basico import (
    load_model,
    get_species,
    get_parameters,
    get_reactions,
    set_species,
    run_time_course,
    get_compartments,
    new_model,
    set_reaction_parameters,
    add_reaction,
    set_reaction,
    set_parameters,
    add_parameter)
from process_bigraph import Step
# from process_bigraph.experiments.parameter_scan import RunProcess
from biosimulators_utils.combine.io import CombineArchiveReader

from biosimulators_processes import CORE
from biosimulators_processes.helpers import calc_num_steps, calc_duration, calc_step_size
from biosimulators_processes.data_model.sed_data_model import MODEL_TYPE
from biosimulators_processes.io import get_model_file_location, FilePath, get_published_t, get_sedml_time_config


class UniformTimeCourse(Step, abc.ABC):

    config_schema = {
        'model': MODEL_TYPE,  # model changes can go here. A dict of dicts of dicts: {'model_changes': {'floating_species': {'t': {'initial_concentration' value'}}}}
        'archive_filepath': 'maybe[string]',
        'working_dirpath': 'maybe[string]',
        'time_config': {
            'duration': 'maybe[float]',  # make these maybes
            'num_steps': 'maybe[float]',
            'step_size': 'maybe[float]'
        }
        # add simulator kwargs
    }

    def __init__(self,
                 config=None,
                 core=CORE,
                 archive_filepath: str = None,
                 sbml_filepath: str = None,
                 working_dirpath: str = None,
                 time_config: Dict[str, Union[int, float]] = None,
                 sed_model_config: Dict = None):
        """Abstract base class which implements the `process_bigraph.Step` interface for ODE-based simulations and simulators. The implementer
            must implement the specific abstract methods defined here.

            Parameters:
                archive_filepath:`optional[str]`: path to either an OMEX/Combine file (`.omex`), or a directory of an "unpacked/extracted" archive containing
                    reports and/or expected results.
                sbml_filepath:`optional[str]`: path to a sbml file.
                working_dirpath:`optional[str]`: path to working directory to unpack archive and/or save outputs.
                time_config:`optional[dict]`: defining duration, num_steps, and step_size (two of the three must be specified).
                sed_model_config:`optional[dict]`: defining model specs as per SED standards.
                    See `biosimulators_processes.data_model.sed_data_model.MODEL_TYPE` for more information.
                    At least `{'model': {'model_source': ...}}` must be defined.
                config:`optional[dict]`: process-bigraph-style configuration.
                core: typesystem used.
        """

        # verify entrypoints
        assert archive_filepath or sbml_filepath and time_config or config, \
            "You must pass either an omex archive filepath, time config and sbml_filepath, or a config dict."

        # extract archive file into working dir if needed
        if archive_filepath and not os.path.isdir(archive_filepath):
            working_dir = working_dirpath or mkdtemp()
            archive_filepath = CombineArchiveReader().run(in_file=archive_filepath, out_dir=working_dir)

        # parse expected results timecourse config
        utc_config = get_sedml_time_config(os.path.join(archive_filepath, 'simulation.sedml')) \
            if archive_filepath else time_config
        assert len(list(utc_config.values())) >= 2, "you must pass two of either: step size, n steps, or duration."

        # parse config
        sbml_fp = sbml_filepath or get_model_file_location(archive_filepath).path
        print(os.path.exists(sbml_fp))
        configuration = config or {'model': {'model_source': sbml_fp}, 'time_config': utc_config}

        # calc/set time params
        self.step_size = utc_config.get('step_size')
        self.num_steps = utc_config.get('num_steps')
        self.duration = utc_config.get('duration')
        if len(list(utc_config.values())) < 3:
            self._set_time_params()

        super().__init__(config=configuration, core=core)

        # set simulator library-specific attributes
        self.simulator = self._set_simulator(sbml_fp)
        self.floating_species_ids = self._get_floating_species_ids()
        self.t = np.linspace(0, self.duration, self.num_steps) if not archive_filepath else get_published_t(archive_filepath)  # create SEDML-SED translator

    def _set_time_params(self):
        if self.step_size and self.num_steps:
            self.duration = calc_duration(self.num_steps, self.step_size)
        elif self.step_size and self.duration:
            self.num_steps = calc_num_steps(self.duration, self.step_size)
        else:
            self.step_size = calc_step_size(self.duration, self.num_steps)

    @abc.abstractmethod
    def _set_simulator(self, sbml_fp: str) -> Any:
        """Load simulator instance with self.config['sbml_filepath']"""
        pass

    @abc.abstractmethod
    def _get_floating_species_ids(self) -> list[str]:
        """Sim specific method"""
        pass

    def inputs(self):
        """For now, none"""
        # return {'duration': 'maybe[int]',
                # 'num_steps': 'maybe[int]',
                # 'step_size': 'maybe[float]'}
        return {}

    def outputs(self):
        return {'time': 'list[float]',
                'floating_species': {spec_id: 'float' for spec_id in self.floating_species_ids}}

    def update(self, inputs=None) -> dict[str, dict[str, list[Any]] | ndarray[Any, dtype[Any]] | ndarray]:
        """Iteratively update over self.floating_species_ids as per the requirements of the simulator library over
            this class' `t` attribute, which is are linearly spaced time-point vectors. Inputs may be passed to override
            anything in this method and/or class in general.
        """

        results = {
            'time': self.t,
            'floating_species': {
                mol_id: []
                for mol_id in self.floating_species_ids}}

        for i, ti in enumerate(self.t):
            start = self.t[i - 1] if ti > 0 else ti
            end = self.t[i]
            timecourse = self._run_simulation(
                start_time=start,
                duration=end)

            # TODO: just return the run time course return

            for mol_id in self.floating_species_ids:
                output = float(self._get_floating_species_concentrations(species_id=mol_id, model=self.simulator))
                results['floating_species'][mol_id].append(output)

        return results

    def run(self, input_state: Dict[str, Union[List[str], Dict[str, List[float]]]] = None, **simulator_kwargs):
        return self.update(inputs=input_state or {})

    @abc.abstractmethod
    def _run_simulation(self, start_time, duration, **simulator_kwargs):
        """Run timecourse simulation as per simulator library requirements"""
        pass

    @abc.abstractmethod
    def _get_floating_species_concentrations(self, species_id: str, model: object = None):
        """Get floating species concentration values as per specific simulator library requirements"""
        pass

    # @abc.abstractmethod
    # def _parse_input_state(self, inputs, **simulator_kwargs):
        """Set species/param values and override constructor settings prior to simulation.
            1. iterate over requested overrides and call a `get` routine over the given class data id attributes like `floating_species_ids`, etc.
            2. update `self.t` if any time overrides are requested by recalculating self.duration, etc.
            3. set any simulator data if any "global" or "local" simulator-specific overrides are requested in the `**simulator_kwargs`
            TODO: finish this.
        """
        # pass


class CopasiStep(UniformTimeCourse):
    config_schema = {
        'model': MODEL_TYPE,  # model changes can go here. A dict of dicts of dicts: {'model_changes': {'floating_species': {'t': {'initial_concentration' value'}}}}
        'archive_filepath': 'maybe[string]',
        'working_dirpath': 'maybe[string]',
        'time_config': {
            'duration': 'maybe[float]',  # make these maybes
            'num_steps': 'maybe[float]',
            'step_size': 'maybe[float]'
        }
        # add simulator kwargs
    }

    def __init__(self,
                 archive_filepath: str = None,
                 sbml_filepath: str = None,
                 time_config: Dict[str, Union[int, float]] = None,
                 config=None,
                 core=CORE):
        super().__init__(archive_filepath=archive_filepath, sbml_filepath=sbml_filepath, time_config=time_config, config=config, core=core)

    def _set_simulator(self, sbml_fp: str) -> CDataModel:
        return load_model(sbml_fp)

    def _get_floating_species_ids(self) -> list[str]:
        species_data = get_species(model=self.simulator)
        assert species_data is not None, "Could not load species ids."
        return species_data.index.tolist()

    def _run_simulation(self, start_time: float, duration: int, **simulator_kwargs):
        return run_time_course(
            start_time=start_time,
            duration=duration,
            automatic=True,
            step_number=self.num_steps,
            update_model=True,
            model=self.simulator,
            **simulator_kwargs)

    def _get_floating_species_concentrations(self, species_id: str, model: object = None):
        return get_species(name=species_id, exact=True, model=self.simulator).concentration[0]

    def update(self, inputs=None) -> dict[str, dict[str, list[Any]] | ndarray[Any, dtype[Any]] | ndarray]:
        tc = run_time_course(start_time=0, duration=self.duration, step_number=self.num_steps - 1, model=self.simulator)
        return {
            'time': self.t,
            'floating_species': {
                mol_id: np.array(list(tc.to_dict().get(mol_id).values()))
                for mol_id in self.floating_species_ids
            }
        }


class AmiciStep(UniformTimeCourse):
    """`config` includes 'model_dir' for model compilation."""
    def __init__(self,
                 archive_filepath: str = None,
                 sbml_filepath: str = None,
                 time_config: Dict[str, Union[int, float]] = None,
                 model_dir: str = None,
                 config=None,
                 core=CORE):
        self.model_dir = model_dir or config.get('model_dir') or mkdtemp()
        super().__init__(archive_filepath=archive_filepath, sbml_filepath=sbml_filepath, time_config=time_config, config=config, core=core)
        self.sbml_model_object = None
        self.simulator = self._set_simulator(sbml_filepath)
        self.simulator.setTimepoints(self.t)

    @staticmethod
    def _set_sbml_model(sbml_fp: str) -> libsbml.Model:
        sbml_reader = libsbml.SBMLReader()
        sbml_doc = sbml_reader.readSBML(sbml_fp)
        return sbml_doc.getModel()

    def _set_simulator(self, sbml_fp: str) -> amici.Model:
        # get and compile libsbml model from fp
        self.sbml_model_object = self._set_sbml_model(sbml_fp)
        model_id = self.config['model'].get('model_id', None) \
            or sbml_fp.split('/')[-1].replace('.', '_').split('_')[0]

        # TODO: integrate # observables=self.config.get('observables'),
        #             # constant_parameters=self.config.get('constant_parameters'),
        #             # sigmas=self.config.get('sigmas'))
        # compile sbml to amici
        sbml_importer = SbmlImporter(sbml_fp)
        sbml_importer.sbml2amici(
            model_name=model_id,
            output_dir=self.model_dir,
            verbose=logging.INFO)
        model_module = import_model_module(model_id, self.model_dir)

        return model_module.getModel()

    def _get_floating_species_ids(self) -> List[str]:
        return list(self.simulator.getObservableIds())

    def _run_simulation(self) -> ReturnDataView:
        sol = self.simulator.getSolver()
        return runAmiciSimulation(solver=sol, model=self.simulator)

    def _get_floating_species_concentrations(self, rdata: ReturnDataView):
        """TODO: Finish this."""
        return dict(zip(
            self.floating_species_ids,
            list(map(
                lambda x: rdata.by_id(f'{x}'),
                self.floating_species_ids))
        ))

    def update(self, inputs=None) -> dict[str, dict[str, list[Any]] | ndarray[Any, dtype[Any]] | ndarray]:
        # TODO: handle changes, if any.
        rdata = self._run_simulation()
        species_data = self._get_floating_species_concentrations(rdata)
        return {
            'time': self.t,
            'floating_species': species_data
        }


class TelluriumStep(UniformTimeCourse):
    def __init__(self,
                 archive_filepath: str = None,
                 sbml_filepath: str = None,
                 time_config: Dict[str, Union[int, float]] = None,
                 config=None,
                 core=CORE):
        super().__init__(archive_filepath=archive_filepath, sbml_filepath=sbml_filepath, time_config=time_config, config=config, core=core)

    def _set_simulator(self, sbml_fp) -> ExtendedRoadRunner:
        return te.loadSBMLModel(sbml_fp)

    def _get_floating_species_ids(self) -> list[str]:
        return self.simulator.getFloatingSpeciesIds()

    def _run_simulation(self, start_time, duration, **simulator_kwargs):
        return self.simulator.simulate(start_time, duration, **simulator_kwargs)

    def _get_floating_species_concentrations(self, species_id: str, model: object = None):
        return self.simulator.getValue(species_id)

    def update(self, inputs=None):
        results = self.simulator.simulate(0, self.duration, self.num_steps)
        pass  # TODO: finish this.


class ODEProcess(dict):
    def __init__(self,
                 address: str,
                 model_fp: str,
                 observables: list[list[str]],
                 step_size: float,
                 duration: float,
                 config=None,
                 core=CORE,
                 **kwargs):
        """
            Kwargs:
                'process_address': 'string',
                'process_config': 'tree[any]',
                'observables': 'list[path]',
                'timestep': 'float',
                'runtime': 'float'
        """
        configuration = config or {}
        if not config:
            configuration['process_address'] = address
            configuration['timestep'] = step_size
            configuration['runtime'] = duration
            configuration['process_config'] = {'model': {'model_source': model_fp}}
            configuration['observables'] = observables
        super().__init__(config=configuration, core=core)

    def run(self, inputs=None):
        input_state = inputs or {}
        return self.update(input_state)
