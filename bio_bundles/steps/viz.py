from typing import Tuple, Callable, Union, List, Dict
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from process_bigraph.composite import Step, Composite, RAMEmitter


def parse_composition_results(composition: Composite) -> Dict[float, Dict[str, Union[Dict, int, float]]]:
    """Return a dictionary in which the outer keys are each member of the timescale, which is parsed
        from the `process_bigraph.Composite.state['global_time']`. Each outer key's children represent the
        data returned from the emitter for ports at each time step.

        Args:
            composition:`process_bigraph.Composite`: Composite instance from which data is being retrieved.

        Returns:
            `Dict[float, Dict[str, Union[Dict, int, float]]]`
            A dictionary of keys: result_values.
    """
    results = composition.gather_results()
    result_vals = list(results.values())[0]
    timescale = [
        float(n)
        for n in range(int(composition.state['global_time']) + 1) # python ranges are not as they appear ;)
    ]
    assert len(timescale) == len(result_vals)  # src check: is there a better way to perform the check?
    return dict(zip(timescale, result_vals))


@dataclass
class PlotLabelConfig:
    title: str
    x_label: str
    y_label: str


class ResultsAnimation:
    def __init__(self,
                 x_data: np.array,
                 y_data: np.array,
                 plot_type_func: Callable,
                 plot_label_config: PlotLabelConfig,
                 **config):
        self.x_data = x_data
        self.y_data = y_data
        self.plot_type_func = plot_type_func
        self.plot_label_config = plot_label_config
        self.config = config

    def _set_axis_limits(self, ax_limit_func, data):
        return ax_limit_func(data.min(), data.max())

    def _prepare_animation(self, t):
        plt.cla()
        self.plot_type_func(
            self.x_data,
            self.y_data,
            color=self.config.get('color', 'blue')
        )

        self._set_axis_limits(plt.xlim, self.x_data)
        self._set_axis_limits(plt.ylim, self.y_data)
        plt.axhline(0, color='grey', lw=0.5)
        plt.axvline(0, color='grey', lw=0.5)
        plt.title(self.plot_label_config.title)
        plt.xlabel(self.plot_label_config.x_label)
        plt.ylabel(self.plot_label_config.y_label)

    def _create_animation_components(self) -> Tuple:
        """
        Creates a matplotlib subplot setup for animation.

        Returns:
            Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]: A tuple containing the figure and axes objects.
        """
        plt.rcParams["animation.html"] = "jshtml"
        plt.rcParams['figure.dpi'] = 150  # TODO: add this to config
        plt.ioff()
        fig, ax = plt.subplots()
        return fig, ax

    def run(self, num_frames: int):
        """
        Creates and runs the animation.

        Args:
            num_frames (int): Number of frames to animate

        Returns:
            matplotlib.animation.FuncAnimation: The animation object.
        """
        fig, ax = self._create_animation_components()
        return FuncAnimation(fig, self._prepare_animation, frames=num_frames)



class CompositionPlotter(Step):

    data: Dict[float, Dict[str, Union[Dict, int, float]]]
    emitter: RAMEmitter
    config_schema = {
        'emitter': 'step',
        'plot_counts': {
            '_type': 'boolean',
            '_default': False
        }
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)

        self.emitter = self.config.get('emitter')
        assert self.emitter is not None, 'You must pass a ram-emitter instance into this plotter'
        self.data = self.emitter.history
        self.species_names = []
        for timestamp, result in self.data.items():
            names = list(result['floating_species_concentrations'].keys())
            for n in names:
                self.species_names.append(n)

        self.species_names = list(set(self.species_names))

    def _parse_data(self, species_names: List):
        fig, axes = plt.subplots(nrows=len(species_names), ncols=2, figsize=(15, 5 * len(species_names)))
        fig.suptitle('Species Counts and Concentrations Over Time')

        # Process each species
        for index, species in enumerate(species_names):
            times = list(data.keys())
            counts = [data[time]['floating_species_counts'][species] for time in times]
            concentrations = [data[time]['floating_species_concentrations'][species] for time in times]

            # Plot counts
            ax1 = axes[index, 0]
            ax1.plot(times, counts, label=f'Counts of {species}')
            ax1.set_title(f'Counts of {species}')
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Counts')
            ax1.legend()

            # Plot concentrations
            ax2 = axes[index, 1]
            ax2.plot(times, concentrations, label=f'Concentration of {species}')
            ax2.set_title(f'Concentration of {species}')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Concentration')
            ax2.legend()

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def inputs(self):
        return {'emitter': 'step'}

    def outputs(self):
        return {}

    def update(self, state):
        return self._parse_data(self.species_names)




class Plotter2d(Step):
    config_schema = {
        'duration': 'any',  # TODO: change this
        'process': 'string',
        'species_context': {
            '_type': 'string',
            '_default': 'concentrations'
        }
    }

    def __init__(self, config=None, core=None):
        super().__init__(config, core)
        context = self.config['species_context']
        self.context_key = f'floating_species_{context}'
        self.species_store = [f'{self.context_key}_store']


    def inputs(self):
        return {
            self.context_key: 'tree[float]'
        }

    def outputs(self):
        return {
            self.context_key: 'tree[float]'
        }

    def update(self, state):
        """print(f'GOt the state: {state}')
        results = state[self.context_key]
        species_names = list(results.keys())
        timescale = list(range(self.config['duration'] + 1))
        for name in species_names:
            y_data = results[name]
            self.plot_output(
                x_data=timescale,
                y_data=y_data,
                species=name,
                x_label='Time',
                y_label=self.context_key
            )
        return {}"""
        print('----', state)
        print('----', self.config['process'])
        return {}


    @classmethod
    def plot_output(
            cls,
            x_data: Union[List, np.ndarray],
            y_data: Union[List, np.ndarray],
            **label_config
    ) -> None:
        """Plot arbitrary output.

            Args:
                x_data: array or list of data for x axis.
                y_data: array or list of data for y axis.
                **label_config: kwargs include--> title, x_label, y_label
        """
        plt.figure(figsize=(8, 5))
        plt.plot(x_data, y_data, marker='o', linestyle='-', color='b', label=label_config.get('species'))
        plt.title(label_config.get('title'))
        plt.xlabel(label_config.get('x_label'))
        plt.ylabel(label_config.get('y_label'))
        plt.legend()
        plt.grid(True)
        plt.show()

    @classmethod
    def plot_single_output(
            cls,
            timescale: Union[List, np.array],
            data: Union[List, np.array],
            species_name: str,
            plot_concentration=True
    ) -> None:
        """Plotting function to plot the output of a SINGLE species' concentrations over a timescale for the output
            of a deterministic time course simulation.

            Args:
                timescale:`Union[List, np.array]`: list containing each time step.
                data:`Union[List, np.array]`: output data mapped to each timescale.
                species_name:`str`: Name of the species that you are plotting.
                plot_concentration:`bool`: Whether you are plotting concentrations for data. Effects
                    plot labels. Defaults to `True`.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(timescale, data, marker='o', linestyle='-', color='b', label=species_name)
        plt.title(f'{species_name} over time')
        plt.xlabel('Time')
        plt.ylabel('Concentration' if plot_concentration else 'Counts')
        plt.legend()
        plt.grid(True)
        plt.show()


