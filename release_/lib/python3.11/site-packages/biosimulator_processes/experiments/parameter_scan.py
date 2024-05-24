from process_bigraph import Composite, Step, pf


class ParameterScan(Step):
    config_schema = {
        'parameters': 'tree'  # this should be from parameter id to range of values
    }

    def __init__(self, config=None):
        super().__init__(config)

    def inputs(self):
        return {
            'parameters': ''
        }

    def outputs(self):
        return {}

    def update(self, inputs):
        return {}


def test_param_scan_copasi():
    initial_sim_state = {
        'parameter_scan': {
            '_type': 'step',
            'address': 'local:!biosimulator_processes.experiments.parameter_scan.ParameterScan',
            'config': {},
            'inputs': {
                'parameters': {}
            },
            'outputs': {
                'simulations': {}
            }
        },
        'copasi': {
            '_type': 'process',
            'address': 'local:copasi',
            'config': {
                'model_file': 'model_files/Caravagna2010.xml'
            },
            'inputs': {
                'floating_species': ['floating_species_store'],
                'model_parameters': ['model_parameters_store'],
                'time': ['time_store'],
                'reactions': ['reactions_store']
            },
            'outputs': {
                'floating_species': ['floating_species_store'],
                'time': ['time_store'],
            }
        },
    }

    # 2. Make the composite:
    workflow = Composite({
        'state': initial_sim_state
    })

    # 3. Run the composite workflow:
    workflow.run(10)

    # 4. Gather and pretty print results
    results = workflow.gather_results()
    print(f'RESULTS: {pf(results)}')


def run_param_scan_cobra():
    instance = {
        'fba': {
            '_type': 'process',
            'address': 'local:cobra',
            'config': {
                'model_file': 'model_files/e_coli_core.xml'
            },
            'inputs': {
                'model': ['model_store'],
                'reaction_bounds': ['reaction_bounds_store'],
                'objective_reaction': ['objective_reaction_store'],
            },
            'outputs': {
                'fluxes': ['fluxes_store'],
                'objective_value': ['objective_value_store'],
                'reaction_dual_values': ['reaction_dual_values_store'],
                'metabolite_dual_values': ['metabolite_dual_values_store'],
                'status': ['status_store'],
            }
        },
        'emitter': {
            '_type': 'step',
            'address': f'local:ram-emitter',
            'config': {
                'inputs_schema': 'tree[any]'
            },
            'inputs': {'data': 'fluxes_store'},
        }
    }

    # make the composite
    workflow = Composite({
        'state': instance
    })

    # run
    workflow.run(1)

    # gather results
    results = workflow.gather_results()
    print(f'RESULTS: {pf(results)}')


if __name__ == '__main__':
    test_param_scan_copasi()
    # test_param_scan_cobra()
