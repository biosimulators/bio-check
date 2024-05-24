from bigraph_viz import plot_bigraph, plot_flow, pf
from bigraph_viz.dict_utils import schema_keys
from sed2.core import register, ports, annotate, Composite, ProcessRegistry
from sed2.processes import sed_process_registry
import numpy as np

schema_keys.extend(['_class', 'config'])
sbml_model_path = 'susceptible_zombie.xml'
sbml_model_description_path = 'susceptible_zombie.csv'


def ex1():
    # SED document serialized
    instance1 = {
        'time_start': 0,
        'time_end': 10,
        'num_points': 50,
        'selection_list': ['time', 'S', 'Z'],
        'model_path': sbml_model_path,
        'curves': {
            'Susceptible': {'x': 'time', 'y': 'S'},
            'Zombie': {'x': 'time', 'y': 'Z'}
        },
        'figure1name': '"Figure1"',
        'sbml_model_from_path': {
            '_class': 'model_path',
            'wires': {
                'path_to_sbml': 'model_path',
                'model': 'model_instance'
            },
        },
        'plot2d': {
            '_class': 'plot2D',
            'wires': {
                'results': 'results',
                'curves': 'curves',
                'name': 'figure1name',
                'figure': 'figure'
            },
            '_depends_on': ['uniform_time_course'],
        },
        'uniform_time_course': {
            '_class': 'uniform_time_course',
            'wires': {
                'model': 'model_instance',
                'time_start': 'time_start',
                'time_end': 'time_end',
                'num_points': 'num_points',
                'selection_list': 'selection_list',
                'results': 'results',
            },
            '_depends_on': ['sbml_model_from_path'],
        },
    }

    sim_experiment = Composite(
        config=instance1,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)
    print(results)


def ex2():
    instance2 = {
        'model_path': sbml_model_path,
        'UTC': '"UTC"',
        'selection_list': ['S', 'Z'],
        'sbml_model_from_path': {
            '_class': 'model_path',
            'wires': {
                'path_to_sbml': 'model_path',
                'model': 'model_instance'
            },
        },
        'steady_state_values': {
            '_class': 'steady_state',
            'wires': {
                'model': 'model_instance',
                # 'time_start': 'time_start',
                # 'time_end': 'time_end',
                # 'num_points': 'num_points',
                'selection_list': 'selection_list',
                'results': 'results',
            },
            '_depends_on': ['sbml_model_from_path']
        },
        'report': {
            '_class': 'report',
            'wires': {
                'results': 'results',
                'title': 'UTC'  # this should be optional
            },
            '_depends_on': ['steady_state_values']
        }
    }

    sim_experiment = Composite(
        config=instance2,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)
    print(results)


def ex3():
    instance3 = {
        'model_path': sbml_model_path,
        'sbml_model_from_path': {
            '_class': 'model_path',
            'wires': {
                'path_to_sbml': 'model_path',
                'model': 'model_instance'
            },
        },
        'element_id': 'Z',
        'element_value': 0.0,
        'model_set_value': {
            '_class': 'set_model',
            'wires': {
                'model_instance': 'model_instance',
                'element_id': 'element_id',
                'value': 'element_value'
            },
            '_depends_on': ['sbml_model_from_path']
        },
        'selection_list': ['S', 'Z'],
        'steady_state_values': {
            '_class': 'steady_state',
            'wires': {
                'model': 'model_instance',
                # 'time_start': 'time_start',
                # 'time_end': 'time_end',
                # 'num_points': 'num_points',
                'selection_list': 'selection_list',
                'results': 'results',
            },
            '_depends_on': ['model_set_value']
        },
        'report': {
            '_class': 'report',
            'wires': {
                'results': 'results',
                'title': 'UTC'  # this should be optional
            },
            '_depends_on': ['steady_state_values']
        }
    }

    sim_experiment = Composite(
        config=instance3,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)


def ex4():
    instance4 = {
        'model_path': sbml_model_path,
        'sbml_model_from_path': {
            '_class': 'model_path',
            'wires': {
                'path_to_sbml': 'model_path',
                'model': 'model_instance'
            },
        },
        'repeated_sim_config': {'Z': list(range(1, 11))},
        'repeated_simulation': {
            '_class': 'repeated_simulation',
            'wires': {
                'model_instance': 'model_instance',
                'config': 'repeated_sim_config',
                'results': 'results',
            },
            '_depends_on': ['sbml_model_from_path']
        },
        'report': {
            '_class': 'report',
            'wires': {
                'results': 'results',
                'title': 'UTC'  # this should be optional
            },
            '_depends_on': ['repeated_simulation']
        }
    }

    sim_experiment = Composite(
        config=instance4,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)


def ex5():
    instance5 = {
        'model_path': sbml_model_path,
        'model_instance': None,
        'UTC': '"UTC"',

        # load the model
        'sbml_model_from_path': {
            '_class': 'model_path',
            'wires': {
                'path_to_sbml': 'model_path',
                'model': 'model_instance'
            },
        },

        # set up trials
        's_trials': 5,

        # a composite process
        'n_dimensional_scan': {
            '_class': 'control:range_iterator:model',
            'wires': {
                'trials': 's_trials',
                'model_instance': 'model_instance',
                'results': 'results',
            },
            '_depends_on': ['..', 'sbml_model_from_path'],

            # state within for_loop
            'time_start': 0,
            'time_end': 10,
            'num_points': 5,

            # process within for_loop
            'uniform_time_course': {
                '_class': 'uniform_time_course',
                'wires': {
                    'model': 'model_instance',  #  ['..', 'model_instance'],  # TODO -- get these to connect
                    'time_start': 'time_start',
                    'time_end': 'time_end',
                    'num_points': 'num_points',
                    'selection_list': 'selection_list',
                    'results': 'results',
                },
                '_depends_on': ['sbml_model_from_path'],
            },
        },
        'results': None,

        # report results
        'report': {
            '_class': 'report',
            'wires': {
                'results': 'results',
                'title': 'UTC'  # this should be optional
            },
            '_depends_on': ['n_dimensional_scan']
        }
    }

    sim_experiment = Composite(
        config=instance5,
        process_registry=sed_process_registry)

    state = {}
    results = sim_experiment.update(state=state)


def run_math():
    math_example = {
        'equation': 'x**2 + a*x + b',
        'vars': {'x': 0},
        'params': {'a': 1, 'b': -6},
        # 'results': {},
        'math_process': {
            '_type': 'math:solve_equation',
            'wires': {
                'equation_str': 'equation',
                'initial_vars': 'vars',
                'parameters': 'params',
                'results': 'results',
            },
        },
        'wires': {
            'results': 'results',
        }
    }

    math_example_experiment = Composite(
        config=math_example,
        process_registry=sed_process_registry)

    results = math_example_experiment.update()

    print(pf(results))



if __name__ == '__main__':
    # ex1()
    # ex2()
    # ex3()
    # ex4()
    # ex5()
    run_math()
