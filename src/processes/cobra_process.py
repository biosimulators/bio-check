"""
COBRA FBA Process
"""


import cobra.io
from cobra.io import read_sbml_model
from process_bigraph import Process, Composite, pf, pp
from biosimulator_processes import CORE


def check_sbml(state, schema, core):
    # Do something to check that the value is a valid SBML file
    valid = cobra.io.sbml.validate_sbml_model(state)  # TODO -- this requires XML
    # valid = cobra.io.load_json_model(value)
    if valid:
        return True
    else:
        return False


bounds_type = {
    'lower_bound': {'_type': 'float', '_default': -1000.0},
    'upper_bound': {'_type': 'float', '_default': 1000.0},
}
bounds_tree_type = {
    '_type': 'tree[bounds]',  # TODO -- make this a dict, to make it only one level deep
}
sbml_type = {
    '_inherit': 'string',
    '_check': check_sbml,
    '_apply': 'set',
}

# register new types
CORE.type_registry.register('bounds', bounds_type)
CORE.type_registry.register('sbml', sbml_type)


class CobraProcess(Process):

    config_schema = {
        'model_file': 'sbml',
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        self.model = read_sbml_model(self.config['model_file'])
        self.reactions = self.model.reactions
        self.metabolites = self.model.metabolites
        self.objective = self.model.objective.to_json()['expression']['args'][0]['args'][1]['name']  # TODO -- fix this in cobra
        self.boundary = self.model.boundary

    def initial_state(self):
        solution = self.model.optimize()
        optimized_fluxes = solution.fluxes

        state = {
            'inputs': {
                'reaction_bounds': {}
            },
            'outputs': {
                'fluxes': {}
            }
        }
        for reaction in self.model.reactions:
            state['inputs']['reaction_bounds'][reaction.id] = {
                'lower_bound': reaction.lower_bound,
                'upper_bound': reaction.upper_bound
            }
            state['outputs']['fluxes'][reaction.id] = optimized_fluxes[reaction.id]
        return state

    def inputs(self):
        return {
            'model': 'sbml',
            'reaction_bounds': {
                reaction.id: 'bounds' for reaction in self.reactions
            },
            'objective_reaction': {
                '_type': 'string',
                '_default': self.objective
            },
        }

    def outputs(self):
        return {
            'fluxes': {
                reaction.id: 'float' for reaction in self.reactions
            },
            'objective_value': 'float',
            'reaction_dual_values': {
                reaction.id: 'float' for reaction in self.reactions
            },
            'metabolite_dual_values': {
                metabolite.id: 'float' for metabolite in self.metabolites
            },
            'status': 'string',
        }


    def update(self, inputs, interval):
        # set reaction bounds
        reaction_bounds = inputs['reaction_bounds']
        for reaction_id, bounds in reaction_bounds.items():
            self.model.reactions.get_by_id(reaction_id).bounds = (bounds['lower_bound'], bounds['upper_bound'])

        # set objective
        # TODO -- look into optlang for specifying objective and constraints
        self.model.objective = self.model.reactions.get_by_id(inputs['objective_reaction'])

        # run solver
        solution = self.model.optimize()

        return {
            'fluxes': solution.fluxes.to_dict(),
            'objective_value': solution.objective_value,
            'reaction_dual_values': solution.reduced_costs.to_dict(),
            'metabolite_dual_values': solution.shadow_prices.to_dict(),
            'status': solution.status,
        }


def test_process():
    CORE.process_registry.register('biosimulator_processes.processes.cobra_process.CobraProcess', CobraProcess)
    instance = {
        'fba': {
            '_type': 'process',
            'address': 'local:!biosimulator_processes.processes.cobra_process.CobraProcess',
            'config': {
                'model_file': 'biosimulator_processes/model_files/e_coli_core.xml'
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
        # insert emitter schema
    }

    # make the composite
    # workflow = Composite({
    #     'state': instance
    # })

    # run
    # workflow.run(1)

    # gather results
    # results = workflow.gather_results()
    # print(f'RESULTS: {pf(results)}')
