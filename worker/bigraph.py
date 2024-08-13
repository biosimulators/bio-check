import os
from uuid import uuid4
from typing import *

from process_bigraph import Step
from process_bigraph.composite import ProcessTypes
try:
    import smoldyn as sm
    from smoldyn._smoldyn import MolecState
except:
    raise ImportError(
        '\nPLEASE NOTE: Smoldyn is not correctly installed on your system which prevents you from ' 
        'using the SmoldynProcess. Please refer to the README for further information '
        'on installing Smoldyn.'
    )


class SmoldynStep(Step):
    config_schema = {
        'model_filepath': 'string',
        'animate': {
            '_type': 'boolean',
            '_default': False
        },
        'duration': 'maybe[integer]',
        'dt': 'maybe[float]',
        'initial_species_counts': 'maybe[tree[float]]',
        'initial_mol_position': 'maybe[list[float]]',  # of particles/molecules
        'initial_mol_state': 'maybe[integer]',

        # TODO: Add a more nuanced way to describe and configure dynamic difcs given species interaction patterns
    }

    def __init__(self, config: Dict[str, Any] = None, core=None):
        """A new instance of `SmoldynProcess` based on the `config` that is passed. The schema for the config to be passed in
            this object's constructor is as follows:

            config_schema = {
                'model_filepath': 'string',  <-- analogous to python `str`
                'animate': 'bool'  <-- of type `bigraph_schema.base_types.bool`

            # TODO: It would be nice to have classes associated with this.
        """
        super().__init__(config, core)

        # specify the model fp for clarity
        self.model_filepath = self.config.get('model_filepath')

        # enforce model filepath passing
        if not self.model_filepath:
            raise ValueError(
                '''
                    The Process configuration requires a Smoldyn model filepath to be passed.
                    Please specify a 'model_filepath' in your instance configuration.
                '''
            )

        # initialize the simulator from a Smoldyn MinE.txt file.
        self.simulation: sm.Simulation = sm.Simulation.fromFile(self.model_filepath)

        # get a list of the simulation species
        species_count = self.simulation.count()['species']
        counts = self.config.get('initial_species_counts')
        self.initial_species_counts = counts
        self.species_names: List[str] = []
        for index in range(species_count):
            species_name = self.simulation.getSpeciesName(index)
            if 'empty' not in species_name.lower():
                self.species_names.append(species_name)
            # if counts is None:
                # self.initial_species_counts[species_name] = species_count[index]

        # sort for logistical mapping to species names (i.e: ['a', 'b', c'] == ['0', '1', '2']
        self.species_names.sort()

        # set default starting position of molecules/particles (assume all)
        self.initial_mol_position = self.config.get('initial_mol_position', [0.0, 0.0, 0.0])
        self.initial_mol_state = self.config.get('initial_mol_state', 0)

        # make species counts of molecules dataset for output
        self.simulation.addOutputData('species_counts')
        # write molcounts to counts dataset at every timestep (shape=(n_timesteps, 1+n_species <-- one for time)): [timestep, countSpec1, countSpec2, ...]
        self.simulation.addCommand(cmd='molcount species_counts', cmd_type='E')

        # make molecules dataset (molecule information) for output
        self.simulation.addOutputData('molecules')
        # write coords to dataset at every timestep (shape=(n_output_molecules, 7)): seven being [timestep, smol_id(species), mol_state, x, y, z, mol_serial_num]
        self.simulation.addCommand(cmd='listmols molecules', cmd_type='E')

        # initialize the molecule ids based on the species names. We need this value to properly emit the schema, which expects a single value from this to be a str(int)
        # the format for molecule_ids is expected to be: 'speciesId_moleculeNumber'
        self.molecule_ids = []
        for i, count in enumerate(species_count):
            for _ in range(count):
                self.molecule_ids.append(str(uuid4()))

        # get the simulation boundaries, which in the case of Smoldyn denote the physical boundaries
        # TODO: add a verification method to ensure that the boundaries do not change on the next step...
        self.boundaries: Dict[str, List[float]] = dict(zip(['low', 'high'], self.simulation.getBoundaries()))

        # create a re-usable counts and molecules type to be used by both inputs and outputs
        self.counts_type = {
            species_name: 'int'
            for species_name in self.species_names
        }

        self.output_port_schema = {
            'species_counts': {
                species_name: 'int'
                for species_name in self.species_names
            },
            'molecules': 'tree[string]'  # self.molecules_type
        }

        # set time if applicable
        self.duration = self.config.get('duration')
        self.dt = self.config.get('dt', self.simulation.dt)

        # set graphics (defaults to False)
        if self.config['animate']:
            self.simulation.addGraphics('opengl_better')

        self._specs = [None for _ in self.species_names]
        self._vals = dict(zip(self.species_names, [[] for _ in self.species_names]))

    def initial_state(self):
        initial_species_state = {}
        initial_mol_state = {}

        initial_mol_counts = {spec_name: self.simulation.getMoleculeCount(spec_name, MolecState.all) for spec_name in self.species_names}
        for species_name, count in initial_mol_counts.items():
            initial_species_state[species_name] = count
            for _ in range(count):
                initial_mol_state[str(uuid4())] = {
                    'coordinates': self.initial_mol_position,
                    'species_id': species_name,
                    'state': self.initial_mol_state
                }

        return {
            'species_counts': self.initial_species_counts or initial_species_state,
            'molecules': initial_mol_state
        }

    def inputs(self):
        return {'model_filepath': 'string'}

    def outputs(self):
        return self.output_port_schema

    def update(self, inputs: Dict) -> Dict:
        # reset the molecules, distribute the mols according to self.boundarieså
        # for name in self.species_names:
        #     self.set_uniform(
        #         species_name=name,
        #         count=inputs['species_counts'][name],
        #         kill_mol=False
        #     )

        # run the simulation for a given interval if specified, otherwise use builtin time
        if self.duration is not None:
            self.simulation.run(stop=self.duration, dt=self.simulation.dt)
        else:
            self.simulation.runSim()

        # get the counts data, clear the buffer
        counts_data = self.simulation.getOutputData('species_counts')

        # get the final counts for the update
        final_count = counts_data[-1]
        # remove the timestep from the list
        final_count.pop(0)

        # get the data based on the commands added in the constructor, clear the buffer
        molecules_data = self.simulation.getOutputData('molecules')

        # create an empty simulation state mirroring that which is specified in the schema
        simulation_state = {
            'species_counts': {},
            'molecules': {}
        }

        # get and populate the species counts
        for index, name in enumerate(self.species_names):
            input_counts = inputs['species_counts'][name]
            simulation_state['species_counts'][name] = int(final_count[index]) - input_counts

        # clear the list of known molecule ids and update the list of known molecule ids (convert to an intstring)
        # self.molecule_ids.clear()
        # for molecule in molecules_data:
            # self.molecule_ids.append(str(uuid4()))

        # get and populate the output molecules
        mols = []
        for index, mol_id in enumerate(self.molecule_ids):
            single_molecule_data = molecules_data[index]
            single_molecule_species_index = int(single_molecule_data[1]) - 1
            mols.append(single_molecule_species_index)
            simulation_state['molecules'][mol_id] = {
                'coordinates': single_molecule_data[3:6],
                'species_id': self.species_names[single_molecule_species_index],
                'state': str(int(single_molecule_data[2]))
            }

        # TODO -- post processing to get effective rates

        return simulation_state

    def set_uniform(
            self,
            species_name: str,
            count: int,
            kill_mol: bool = True
    ) -> None:
        """Add a distribution of molecules to the solution in
            the simulation memory given a higher and lower bound x,y coordinate. Smoldyn assumes
            a global boundary versus individual species boundaries. Kills the molecule before dist if true.

            TODO: If pymunk expands the species compartment, account for
                  expanding `highpos` and `lowpos`. This method should be used within the body/logic of
                  the `update` class method.

            Args:
                species_name:`str`: name of the given molecule.
                count:`int`: number of molecules of the given `species_name` to add.
                kill_mol:`bool`: kills the molecule based on the `name` argument, which effectively
                    removes the molecule from simulation memory.
        """
        # kill the mol, effectively resetting it
        if kill_mol:
            self.simulation.runCommand(f'killmol {species_name}')

        # TODO: eventually allow for an expanding boundary ie in the configuration parameters (pymunk?), which is defies the methodology of smoldyn

        # redistribute the molecule according to the bounds
        self.simulation.addSolutionMolecules(
            species=species_name,
            number=count,
            highpos=self.boundaries['high'],
            lowpos=self.boundaries['low']
        )


class SimulariumStep(Step):
    pass


# register processes

CORE = ProcessTypes()
REGISTERED_PROCESSES = [
    ('smoldyn_step', SmoldynStep),
    ('simularium_step', SimulariumStep)
]
for process_name, process_class in REGISTERED_PROCESSES:
    try:
        CORE.process_registry.register(process_name, process_class)
    except Exception as e:
        print(f'{process_name} could not be registered because {str(e)}')

