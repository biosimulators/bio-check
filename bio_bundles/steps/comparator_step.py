import pandas as pd
from process_bigraph import Step

from biosimulators_processes import CORE
from biosimulators_processes.api.compare import generate_utc_species_comparison


class UtcComparator(Step):
    config_schema = {
        'simulators': 'list[string]',
        'comparison_id': 'string',
        'comparison_method': 'string',
        'include_output_data': {
            '_default': True,
            '_type': 'boolean'
        },

    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.simulators = self.config['simulators']
        self.include_output = self.config['include_output_data']
        self.comparison_id = self.config.get('comparison_id', 'utc-comparison')
        self.comparison_method = self.config.get('comparison_method')

    def inputs(self): 
        port_schema = {
            f'{simulator}_floating_species': 'tree[float]'
            for simulator in self.simulators 
        }
        port_schema['time'] = 'list[float]'
        return port_schema
    
    def outputs(self):
        return {
            'results': 'tree[float]',  # ie: {spec_id: {sim_name: outputarray}}
            'id': 'string'
        }
        
    def update(self, inputs):
        # TODO: more dynamically infer this. Perhaps use libsbml?
        species_names = list(inputs['copasi_floating_species'].keys())
        _data = dict(zip(species_names, {}))
        results = {
            'results': _data,
            'id': self.comparison_id
        }

        for name in species_names:
            outputs = [
                inputs[f'{simulator}_floating_species'][name]
                for simulator in self.simulators]

            shapes = [v.shape[0] for v in outputs]
            min_len = min(shapes)
            for i, val in enumerate(outputs):
                if val.shape[0] > min_len:
                    outputs.pop(i)
                    outputs.insert(i, val[:min_len])

                # if 'copasi' not in self.simulators[i].lower():
                #     outputs.pop(i)
                #     outputs.insert(i, val[:600])

            comparison = generate_utc_species_comparison(
                outputs=outputs,
                simulators=self.simulators,
                species_name=name)

            comparison_data = comparison.to_dict() if isinstance(comparison, pd.DataFrame) else comparison

            if self.include_output:
                comparison_data['output_data'] = inputs

            results['results'][name] = comparison_data

        return results
