"""
Dynamic FBA simulation
======================

Process for a pluggable dFBA simulation.
"""

import numpy as np
import warnings

import cobra
from cobra.io import load_model
from process_bigraph import Process, Composite

from biosimulators_processes import CORE
from biosimulators_processes.viz.plot import plot_time_series, plot_species_distributions_to_gif


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cobra.util.solver")
warnings.filterwarnings("ignore", category=FutureWarning, module="cobra.medium.boundary_types")


# create new types
def apply_non_negative(schema, current, update, core):
    new_value = current + update
    return max(0, new_value)


def check_sbml(state, schema, core):
    # Do something to check that the value is a valid SBML file
    valid = cobra.io.sbml.validate_sbml_model(state)  # TODO -- this requires XML
    # valid = cobra.io.load_json_model(value)
    if valid:
        return True
    else:
        return False


positive_float = {
    '_type': 'positive_float',
    '_inherit': 'float',
    '_apply': apply_non_negative
}
CORE.register('positive_float', positive_float)

bounds_type = {
    'lower': 'maybe[float]',
    'upper': 'maybe[float]'
}
CORE.register_process('bounds', bounds_type)

sbml_type = {
    '_inherit': 'string',
    '_check': check_sbml,
    '_apply': 'set',
}

# register new types
CORE.type_registry.register('sbml', sbml_type)

# TODO -- can set lower and upper bounds by config instead of hardcoding
MODEL_FOR_TESTING = load_model('textbook')
# MODEL_FOR_TESTING.reactions.EX_o2_e.lower_bound = -2  # Limiting oxygen uptake
# MODEL_FOR_TESTING.reactions.ATPM.lower_bound = 1     # Setting lower bound for ATP maintenance
# MODEL_FOR_TESTING.reactions.ATPM.upper_bound = 1     # Setting upper bound for ATP maintenance


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


class DynamicFBA(Process):
    """
    Performs dynamic FBA.

    Parameters:
    - model: The metabolic model for the simulation.
    - kinetic_params: Kinetic parameters (Km and Vmax) for each substrate.
    - biomass_reaction: The identifier for the biomass reaction in the model.
    - substrate_update_reactions: A dictionary mapping substrates to their update reactions.
    - biomass_identifier: The identifier for biomass in the current state.

    TODO -- check units
    """

    config_schema = {
        'model_file': 'string',
        'model': 'Any',
        'kinetic_params': 'map[tuple[float,float]]',
        'biomass_reaction': {
            '_type': 'string',
            '_default': 'Biomass_Ecoli_core'
        },
        'substrate_update_reactions': 'map[string]',
        'biomass_identifier': 'string',
        'bounds': 'map[bounds]',
    }

    def __init__(self, config, core):
        super().__init__(config, core)

        if self.config['model_file'] == 'TESTING':
            self.model = MODEL_FOR_TESTING
        elif not 'xml' in self.config['model_file']:
            # use the textbook model if no model file is provided
            self.model = load_model(self.config['model_file'])
        elif isinstance(self.config['model_file'], str):
            self.model = cobra.io.read_sbml_model(self.config['model_file'])
        else:
            # error handling
            raise ValueError('Invalid model file')

        for reaction_id, bounds in self.config['bounds'].items():
            if bounds['lower'] is not None:
                self.model.reactions.get_by_id(reaction_id).lower_bound = bounds['lower']
            if bounds['upper'] is not None:
                self.model.reactions.get_by_id(reaction_id).upper_bound = bounds['upper']

    def inputs(self):
        return {
            'substrates': 'map[positive_float]'
        }

    def outputs(self):
        return {
            'substrates': 'map[positive_float]'
        }

    # TODO -- can we just put the inputs/outputs directly in the function?
    def update(self, state, interval):
        substrates_input = state['substrates']

        for substrate, reaction_id in self.config['substrate_update_reactions'].items():
            Km, Vmax = self.config['kinetic_params'][substrate]
            substrate_concentration = substrates_input[substrate]
            uptake_rate = Vmax * substrate_concentration / (Km + substrate_concentration)
            self.model.reactions.get_by_id(reaction_id).lower_bound = -uptake_rate

        substrate_update = {}

        solution = self.model.optimize()
        if solution.status == 'optimal':
            current_biomass = substrates_input[self.config['biomass_identifier']]
            biomass_growth_rate = solution.fluxes[self.config['biomass_reaction']]
            substrate_update[self.config['biomass_identifier']] = biomass_growth_rate * current_biomass * interval

            for substrate, reaction_id in self.config['substrate_update_reactions'].items():
                flux = solution.fluxes[reaction_id] * current_biomass * interval
                old_concentration = substrates_input[substrate]
                new_concentration = max(old_concentration + flux, 0)  # keep above 0
                substrate_update[substrate] = new_concentration - old_concentration
                # TODO -- assert not negative?
        else:
            # Handle non-optimal solutions if necessary
            # print('Non-optimal solution, skipping update')
            for substrate, reaction_id in self.config['substrate_update_reactions'].items():
                substrate_update[substrate] = 0

        return {
            'substrates': substrate_update,
        }


# register the process
CORE.register_process('DynamicFBA', DynamicFBA)


# Helper functions to get specs and states
def dfba_config(
        model_file='textbook',
        kinetic_params=None,
        biomass_reaction='Biomass_Ecoli_core',
        substrate_update_reactions=None,
        biomass_identifier='biomass',
        bounds=None
):
    if substrate_update_reactions is None:
        substrate_update_reactions = {
            'glucose': 'EX_glc__D_e',
            'acetate': 'EX_ac_e'}
    if bounds is None:
        bounds = {
            'EX_o2_e': {'lower': -2, 'upper': None},
            'ATPM': {'lower': 1, 'upper': 1}}
    if kinetic_params is None:
        kinetic_params = {
            'glucose': (0.5, 1),
            'acetate': (0.5, 2)}
    return {
        'model_file': model_file,
        'kinetic_params': kinetic_params,
        'biomass_reaction': biomass_reaction,
        'substrate_update_reactions': substrate_update_reactions,
        'biomass_identifier': biomass_identifier,
        'bounds': bounds
    }


def get_single_dfba_spec(mol_ids=None, path=None, i=None, j=None):
    """
    Constructs a configuration dictionary for a dynamic FBA process with optional path indices.

    This function builds a process specification for use with a dynamic FBA system. It allows
    specification of substrate molecule IDs and optionally appends indices to the paths for those substrates.

    Parameters:
        mol_ids (list of str, optional): List of molecule IDs to include in the process. Defaults to
                                         ['glucose', 'acetate', 'biomass'].
        path (list of str, optional): The base path to prepend to each molecule ID. Defaults to ['..', 'fields'].
        i (int, optional): The first index to append to the path for each molecule, if not None.
        j (int, optional): The second index to append to the path for each molecule, if not None.

    Returns:
        dict: A dictionary containing the process type, address, configuration, and paths for inputs
              and outputs based on the specified molecule IDs and indices.
    """
    if path is None:
        path = ['..', 'fields']
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']

    # Function to build the path with optional indices
    def build_path(mol_id):
        base_path = path + [mol_id]
        if i is not None:
            base_path.append(i)
        if j is not None:
            base_path.append(j)
        return base_path

    return {
        '_type': 'process',
        'address': 'local:DynamicFBA',
        'config': dfba_config(),
        'inputs': {
            'substrates': {mol_id: build_path(mol_id) for mol_id in mol_ids}
        },
        'outputs': {
            'substrates': {mol_id: build_path(mol_id) for mol_id in mol_ids}
        }
    }


def get_spatial_dfba_spec(n_bins=(5, 5), mol_ids=None):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    dfba_processes_dict = {}
    for i in range(n_bins[0]):
        for j in range(n_bins[1]):
            dfba_processes_dict[f'[{i},{j}]'] = get_single_dfba_spec(mol_ids=mol_ids, path=['..', 'fields'], i=i, j=j)
    return dfba_processes_dict


def get_spatial_dfba_state(
        n_bins=(5, 5),
        mol_ids=None,
        initial_max=None,
):
    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    if initial_max is None:
        initial_max = {
            'glucose': 20,
            'acetate': 0,
            'biomass': 0.1
        }
    # initial_fields = {
    #     mol_id: np.random.uniform(low=0, high=initial_max[mol_id], size=n_bins[1], n_bins[0]))
    #     for mol_id in mol_ids}
    initial_fields = {
        mol_id: np.random.uniform(low=0, high=initial_max[mol_id], size=n_bins)
        for mol_id in mol_ids}

    return {
        'fields': {
            '_type': 'map',
            '_value': {
                '_type': 'array',
                '_shape': n_bins,
                '_data': 'positive_float'
            },
            **initial_fields,
        },
        'spatial_dfba': get_spatial_dfba_spec(n_bins=n_bins, mol_ids=mol_ids)
    }


def run_dfba_spatial(
        total_time=60,
        n_bins=(3, 3),  # TODO -- why can't do (5, 10)??
        mol_ids=None,
):

    if mol_ids is None:
        mol_ids = ['glucose', 'acetate', 'biomass']
    composite_state = get_spatial_dfba_state(
        n_bins=n_bins,
        mol_ids=mol_ids,
    )

    # make the composite
    print('Making the composite...')
    sim = Composite({
        'state': composite_state,
        'emitter': {'mode': 'all'}
    }, core=CORE)

    # save the document
    sim.save(filename='spatial_dfba.json', outdir='out')

    # simulate
    print('Simulating...')
    sim.update({}, total_time)

    # gather results
    dfba_results = sim.gather_results()
    # print(dfba_results)

    print('Plotting results...')
    # plot timeseries
    plot_time_series(
        dfba_results,
        coordinates=[(0, 0), (1, 1), (2, 2)],
        out_dir='out',
        filename='dfba_timeseries.png',
    )

    # make video
    plot_species_distributions_to_gif(
        dfba_results,
        out_dir='out',
        filename='dfba_results.gif',
        title='',
        skip_frames=1
    )





# if __name__ == '__main__':
    # run_dfba_spatial()