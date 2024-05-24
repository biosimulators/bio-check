"""
Biosimulator process for Copasi/Basico
"""


from basico import (
                load_model,
                get_species,
                get_parameters,
                get_reactions,
                set_species,
                run_time_course,
                get_compartments,
                model_info
            )
from process_bigraph import Process, Composite, pf


class CopasiProcess(Process):
    config_schema = {'model_file': 'string'}

    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        # Load the single cell model into Basico
        self.copasi_model_object = load_model(self.config['model_file'])

        # Get the species (floating only)  TODO: add boundary species
        self.floating_species_list = get_species(model=self.copasi_model_object).index.tolist()
        self.floating_species_initial = get_species(model=self.copasi_model_object)['concentration'].tolist()

        # Get the list of parameters and their values
        self.model_parameters_list = get_parameters(model=self.copasi_model_object).index.tolist()
        self.model_parameter_values = get_parameters(model=self.copasi_model_object)['initial_value'].tolist()

        # Get a list of reactions
        self.reaction_list = get_reactions(model=self.copasi_model_object).index.tolist()

        # Get a list of compartments
        self.compartments_list = get_compartments(model=self.copasi_model_object).index.tolist()

    def initial_state(self):
        floating_species_dict = dict(zip(self.floating_species_list, self.floating_species_initial))
        model_parameters_dict = dict(zip(self.model_parameters_list, self.model_parameter_values))
        return {
            'time': 0.0,
            'floating_species': floating_species_dict,
            'model_parameters': model_parameters_dict
        }

    def inputs(self):
        floating_species_type = {
            species_id: {
                '_type': 'float',
                '_apply': 'set',
            } for species_id in self.floating_species_list
        }
        return {
            'time': 'float',
            'floating_species': floating_species_type,
            'model_parameters': {
                param_id: 'float' for param_id in self.model_parameters_list},
            'reactions': {
                reaction_id: 'float' for reaction_id in self.reaction_list},
        }

    def outputs(self):
        floating_species_type = {
            species_id: {
                '_type': 'float',
                '_apply': 'set',
            } for species_id in self.floating_species_list
        }
        return {
            'time': 'float',
            'floating_species': floating_species_type
        }

    def update(self, inputs, interval):

        # set copasi values according to what is passed in states
        for cat_id, value in inputs['floating_species'].items():
            set_species(name=cat_id, initial_concentration=value, model=self.copasi_model_object)

        # run model for "interval" length; we only want the state at the end
        timecourse = run_time_course(
            start_time=inputs['time'],
            duration=interval,
            intervals=1,
            update_model=True,
            model=self.copasi_model_object)

        # extract end values of concentrations from the model and set them in results
        results = {'time': interval}
        results['floating_species'] = {
            mol_id: float(get_species(name=mol_id, exact=True, model=self.copasi_model_object).concentration[0])
            for mol_id in self.floating_species_list}

        return results


def test_process():
    # 1. Define the sim state schema:
    initial_sim_state = {
        'copasi': {
            '_type': 'process',
            'address': 'local:copasi',
            'config': {
                'model_file': 'biosimulator_processes/model_files/Caravagna2010.xml'
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
        # TODO: Add emitter schema
        'emitter': {
            '_type': 'step',
            'address': 'local:ram-emitter',
            'config': {
                'emit': {
                    'floating_species': 'tree[float]',
                    'time': 'float',
                },
            },
            'inputs': {
                'floating_species': ['floating_species_store'],
                'time': ['time_store'],
            }

        }
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
