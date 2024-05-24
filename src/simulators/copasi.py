from COPASI import CDataModel
from basico import *
from pandas import DataFrame


def load_copasi_model(
        sbml_fp: str,
        model_changes: dict = None,  # nested arbitrarily
        species_context: str = "concentrations",
        **config):
    model_source = sbml_fp

    copasi_model_object = load_model(model_source)

    # handle context of species output
    species_context_key = f'floating_species_{species_context}'
    use_counts = 'concentrations' in species_context_key

    # Get a list of reactions
    set_reaction_changes()
    reactions = get_reactions(model=copasi_model_object)
    self.reaction_list = reactions.index.tolist() if reactions is not None else []
    # if not self.reaction_list:
    # raise AttributeError('No reactions could be parsed from this model. Your model must contain reactions to run.')

    # Get the species (floating only)  TODO: add boundary species
    set_species_changes()
    species_data = get_species(model=self.copasi_model_object)
    self.floating_species_list = species_data.index.tolist()
    self.floating_species_initial = species_data.particle_number.tolist() \
        if self.use_counts else species_data.concentration.tolist()

    # Get the list of parameters and their values (it is possible to run a model without any parameters)
    set_global_param_changes()
    model_parameters = get_parameters(model=self.copasi_model_object)
    self.model_parameters_list = model_parameters.index.tolist() \
        if isinstance(model_parameters, DataFrame) else []
    self.model_parameters_values = model_parameters.initial_value.tolist() \
        if isinstance(model_parameters, DataFrame) else []

    # Get a list of compartments
    self.compartments_list = get_compartments(model=self.copasi_model_object).index.tolist()

    # ----SOLVER: Get the solver (defaults to deterministic)
    self.method = self.config['method']


def set_reaction_changes(model_object: CDataModel, reaction_changes: list[dict]):
    existing_reactions = get_reactions(model=model_object)
    existing_reaction_names = existing_reactions.index.tolist() if existing_reactions is not None else []
    if reaction_changes:
        for reaction_change in reaction_changes:
            reaction_name: str = reaction_change['reaction_name']
            param_changes: list[dict[str, float]] = reaction_change['parameter_changes']
            scheme_change: str = reaction_change.get('reaction_scheme')
            # handle changes to existing reactions
            if param_changes:
                for param_name, param_change_val in param_changes:
                    set_reaction_parameters(param_name, value=param_change_val, model=model_object)
            if scheme_change:
                set_reaction(name=reaction_name, scheme=scheme_change, model=model_object)
            # handle new reactions
            if reaction_name not in existing_reaction_names and scheme_change:
                add_reaction(reaction_name, scheme_change, model=model_object)


def set_species_changes(model_object: CDataModel, species_changes: list[dict]):
    """Shaped like:
        {species_id: {species_param_type (ie initial_concentration): param_type_val}}
    """
    if species_changes:
        for species_change in species_changes:
            if isinstance(species_change, dict):
                species_name = species_change.pop('name')
                changes_to_apply = {}
                for spec_param_type, spec_param_value in species_change.items():
                    if spec_param_value:
                        changes_to_apply[spec_param_type] = spec_param_value
                set_species(**changes_to_apply, model=model_object)


def set_global_param_changes(model_object: CDataModel, param_changes: list[dict]):
    """Shaped like:
            param_changes = {param_name: {param_val_type (ie initial_concentration): param_val_type_value}}
    """
    if param_changes:
        for param_change in param_changes:
            param_name = param_change.pop('name')
            for param_type, param_value in param_change.items():
                if not param_value:
                    param_change.pop(param_type)
                # handle changes to existing params
                set_parameters(name=param_name, **param_change, model=model_object)
                # set new params
                global_params = get_parameters(model=model_object)
                if global_params:
                    existing_global_parameters = global_params.index
                    if param_name not in existing_global_parameters:
                        assert param_change.get('initial_concentration') is not None, "You must pass an initial_concentration value if adding a new global parameter."
                        add_parameter(name=param_name, **param_change, model=model_object)


