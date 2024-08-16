import numpy as np
from typing import *

from simulariumio import TrajectoryData, JsonWriter, BinaryWriter
from simulariumio.filters import TranslateFilter
from simulariumio.smoldyn import SmoldynData, SmoldynConverter


# --agent display --

def calculate_agent_radius(m: float, rho: float, scaling_factor: float = 10**(-2)) -> float:
    """Calculate the radius of an agent given its molecular mass and density. Please note: the molecular mass
        of MinE is 11000 Da with a protein density of 1.35 g/cm^3 (1350 kg/m^3).

        Args:
            m:`float`: the molecular mass of the given agent/particle (Daltons).
            rho:`float`: the density of the given agent/particle (kg/m^3).
            scaling_factor:`float`: tiny number by which to scale the output measurement. Defaults to
                `10**(-2)`, which effectively converts from nm to cm.

        Returns:
            `float`: radius of the given agent.
    """
    dalton_to_kg = 1.66053906660e-27  # Conversion factor from Daltons to kilograms
    m_kg = m * dalton_to_kg  # Convert mass to kilograms
    radius_m = ((3 * m_kg) / (4 * np.pi * rho)) ** (1 / 3)  # Calculate radius in meters
    radius_nm = radius_m * 1e9  # Convert radius to nanometers
    return radius_nm * scaling_factor


def generate_agent_params(
        species_names: List[str],
        global_density: Optional[float] = None,
        basis_m: Optional[int] = None,
        model_fp: Optional[str] = None,
        **config
) -> Dict:
    """Generate a dictionary of agent parameters for the purpose of simulation input configuration which define the
        molecular mass and density inherent to a given agent based species names. We cannot call the species
        names from the smoldyn model file here directly, so you MUST pass them in here.

        Args:
            species_names:`List[str]`: List of species names which coorespond to the species names
                in the relative smoldyn model file being simulated.
            global_density:`Optional[float]`: Density by which all agent densities are set. NOTE: this value is
              required if not passing explicit agent configs (**config). Defaults to `None`.
            basis_m:`Optional[int]`: Upper bound value of the molecular mass by which to set the basis for the
              randomization of agent molecular masses in the `randomize_mass` function which takes in a basis
              integer and returns a random integer between 0 and that number. NOTE: this value is required
              if not passing explicit agent configs (**config). Defaults to `None`.
            model_fp:`Optional[str]`: path to a smoldyn model file. Defaults to `None`.
        Keyword Args:
            <AGENT_NAME>:`dict`: an agent name (which should match that which is returned by
                smoldyn.Simulation.getSpeciesName()) as the definition (for example: `MinE=`) and a dictionary with
                `'molecular_mass'` and `'density'` as the keys.

    """
    params = {}
    if not config and not global_density and not basis_m:
        raise ValueError(
            f'You must pass either keyword arguments where the keyword is the agent name and the value is a dict defining molecular_mass and density OR a density AND basis molecular mass.'
        )
    if not config:
        for name in species_names:
            agent_config = config.get(f'{name}')
            if agent_config:
                mass = agent_config['molecular_mass']
                density = agent_config['density']
            else:
                mass = randomize_mass(basis_m)
                density = global_density
            params[name] = {
                'density': density,
                'molecular_mass': mass
            }
    else:
        for k in config.keys():
            params[k] = config[k]
    return params


def randomize_mass(origin: float) -> int:
    return np.random.randint(int(origin))


# -- data composition --

def translate_data_object(
    data: SmoldynData,
    box_size: float,
    n_dim=3,
    translation_magnitude: Optional[Union[int, float]] = None
) -> TrajectoryData:
    """Translate the data object's data if the coordinates are all positive to center the data in the
           simularium viewer.

           Args:
               data:`SmoldynData`: configured simulation output data instance.
               box_size: size of the simularium viewer box.
               n_dim: n dimensions of the simulation output. Defaults to `3`.
               translation_magnitude: magnitude by which to translate and filter. Defaults to `-box_size / 2`.

           Returns:
               `TrajectoryData`: translated data object instance.
       """
    translation_magnitude = translation_magnitude or -box_size / 2
    return SmoldynConverter(data).filter_data(
        [TranslateFilter(translation_per_type={}, default_translation=translation_magnitude * np.ones(n_dim))]
    )


# -- io --

def write_simularium_file(
    data: Union[SmoldynData, TrajectoryData],
    simularium_fp: str,
    json: bool = True,
    validate: bool = True
) -> None:
    """Takes in either a `SmoldynData` or `TrajectoryData` instance and saves a simularium file based on it
        with the name of `simularium_filename`.

        Args:
            data(:obj:`Union[SmoldynData, TrajectoryData]`): data object to save.
            simularium_fp(:obj:`str`): name by which to save the new simularium file.
            json(:obj:`bool`): exports simularium file in JSON format if true; exports in binary if false. Defaults
                to `False` for optimization's sake.
            validate(:obj:`bool`): whether to call the wrapped method using `validate_ids=True`. Defaults
                to `True`.
    """
    print(f'Writing simularium file "{simularium_fp}"')
    if json:
        writer = JsonWriter()
    else:
        writer = BinaryWriter()
    return writer.save(trajectory_data=data, output_path=simularium_fp, validate_ids=validate)
