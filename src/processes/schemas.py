SED_MODEL_TYPE = {
    'model_id': 'string',
    'model_source': 'string',  # TODO: add antimony support here.
    'model_language': {
        '_type': 'string',
        '_default': 'sbml'
    },
    'model_name': {
        '_type': 'string',
        '_default': 'composite_process_model'
    },
    'model_changes': {
        'species_changes': 'maybe[tree[string]]',
        'global_parameter_changes': 'maybe[tree[string]]',
        'reaction_changes': 'maybe[tree[string]]'
    },
    'model_units': 'maybe[tree[string]]'
}
