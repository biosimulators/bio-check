from abc import ABC, abstractmethod
from typing import *

from process_bigraph import Step, Composite
from src.processes import CORE


# process-bigraph classes
class SimulatorComparisonStep(ABC, Step):
    config_schema = {
        'model_entrypoint': 'string',  # one of: OMEX fp, SBML model path, CellML/other.
        'duration': 'integer',
        'simulators': 'union[tree[string], list[string]]'  # either a dict of simulator: version or a list of [simulator] for default versions.
    }

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        source = config['model_entrypoint']
        assert '/' in source, "You must enter a sbml file path."

    def inputs(self):
        return {}

    def outputs(self):
        floating_species_type = {
            species_id: {
                '_type': 'float',
                '_apply': 'set'}
            for species_id in self.floating_species_list}

        return {
            'time': 'float',
            self.species_context_key: floating_species_type}

    def update(self, state):
        results = self._run_composition()
        output = {'comparison_data': results}
        return output

    @abstractmethod
    def _run_composition(self, comp: Composite) -> Dict:
        pass


class OdeComparisonStep(SimulatorComparisonStep):
    """TODO: Use the spec's entrypoint arguments as config attributes here!!!

        1. take in a dur, model/archive, simulation, comparison_method, ground_truth, and more (optional simulators)
        2. with the specified entrypoint, iterate over simulators and
            load model instances from tellurium. copasi, and amici, respectively.
        3. in constructor, create mapping of loaded simulators to their names.
        4. Create 3 seperate methods for updating/setting: tellurium_update, copasi_update, amici_update
        5. Call the three simulator methods in parallel for a "dur" on update.
        6. Return just the df from the comparison step via the output ports.
        7. Closer to the API, make a function that calls this step and uses its outputs as one of the
            parameters to instantiate and return the `ComparisonMatrix`.
    """

    def __init__(self, config=None, core=CORE):
        super().__init__(config, core)
        self.model_source = self.config['model_entrypoint']
        self.duration = self.config['duration']


        model_fp = self.model_source if not self.model_source.startswith('BIO') else fetch_biomodel_sbml_file(self.model_source, save_dir=mkdtemp())

        self.document = {
            'copasi_simple': {
                '_type': 'process',
                'address': 'local:copasi',
                'config': {'model': {'model_source': model_fp}},
                'inputs': {'floating_species_concentrations': ['copasi_simple_floating_species_concentrations_store'],
                           'model_parameters': ['model_parameters_store'],
                           'time': ['time_store'],
                           'reactions': ['reactions_store']},
                'outputs': {'floating_species_concentrations': ['copasi_simple_floating_species_concentrations_store'],
                            'time': ['time_store']}
            },
            'amici_simple': {
                '_type': 'process',
                'address': 'local:amici',
                'config': {'model': {'model_source': model_fp}},
                'inputs': {
                    'floating_species_concentrations': ['amici_simple_floating_species_concentrations_store'],
                    'model_parameters': ['model_parameters_store'],
                    'time': ['time_store'],
                    'reactions': ['reactions_store']},
                'outputs': {
                    'floating_species_concentrations': ['amici_simple_floating_species_concentrations_store'],
                    'time': ['time_store']}
            },
            'emitter': {
                '_type': 'step',
                'address': 'local:ram-emitter',
                'config': {
                    'emit': {
                        'copasi_simple_floating_species_concentrations': 'tree[float]',
                        'amici_simple_floating_species_concentrations': 'tree[float]',
                        'tellurium_simple_floating_species_concentrations': 'tree[float]',
                        'time': 'float'
                    }
                },
                'inputs': {
                    'copasi_simple_floating_species_concentrations': ['copasi_simple_floating_species_concentrations_store'],
                    'amici_simple_floating_species_concentrations': ['amici_simple_floating_species_concentrations_store'],
                    'tellurium_simple_floating_species_concentrations': ['tellurium_simple_floating_species_concentrations_store'],
                    'time': ['time_store']
                }
            },
            'tellurium_simple': {
                '_type': 'process',
                'address': 'local:tellurium',
                'config': {'model': {'model_source': model_fp}},
                'inputs': {'floating_species_concentrations': ['tellurium_simple_floating_species_concentrations_store'],
                           'model_parameters': ['model_parameters_store'],
                           'time': ['time_store'],
                           'reactions': ['reactions_store']},
                'outputs': {'floating_species_concentrations': ['tellurium_simple_floating_species_concentrations_store'],
                            'time': ['time_store']}}}

    # TODO: Do we need this?
    def inputs(self):
        return {}

    def outputs(self):
        return {'comparison_data': 'tree[any]'}

    def update(self, state):
        comp = self._generate_composition()
        results = self._run_composition(comp)
        output = {'comparison_data': results}
        return output

    def _generate_composition(self) -> Composite:
        return Composite(config={'state': self.document}, core=CORE)

    def _run_composition(self, comp: Composite) -> Dict:
        comp.run(self.duration)
        return comp.gather_results()