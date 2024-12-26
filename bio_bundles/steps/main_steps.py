import logging
import os
import re
import uuid
from abc import abstractmethod
from logging import warn
from uuid import uuid4
from typing import *

from process_bigraph import Step, Process
from process_bigraph.composite import Emitter, ProcessTypes
from pymongo import ASCENDING, MongoClient
from pymongo.database import Database
from simulariumio import InputFileData, UnitData, DisplayData, DISPLAY_TYPE
from simulariumio.smoldyn import SmoldynData

from bio_bundles.data_generator import SBML_EXECUTORS
from bio_bundles.io import get_sbml_species_mapping
from bio_bundles.simularium_utils import translate_data_object, write_simularium_file, calculate_agent_radius

try:
    import smoldyn as sm
    from smoldyn._smoldyn import MolecState
except:
    print(
        '\nPLEASE NOTE: Smoldyn is not correctly installed on your system which prevents you from ' 
        'using the SmoldynProcess. Please refer to the README for further information '
        'on installing Smoldyn.'
    )

HISTORY_INDEXES = [
    'data.time',
    [('experiment_id', ASCENDING),
     ('data.time', ASCENDING),
     ('_id', ASCENDING)],
]
CONFIGURATION_INDEXES = [
    'experiment_id',
]
SECRETS_PATH = 'secrets.json'


# -- emitters -- #

class MongoDatabaseEmitter(Emitter):
    client_dict: Dict[int, MongoClient] = {}
    config_schema = {
        'connection_uri': 'string',
        'experiment_id': 'maybe[string]',
        'emit_limit': {
            '_type': 'integer',
            '_default': 4000000
        },
        'database': 'maybe[string]'
    }

    @classmethod
    def create_indexes(cls, table: Any, columns: List[Any]) -> None:
        """Create the listed column indexes for the given DB table."""
        for column in columns:
            table.create_index(column)

    def __init__(self, config, core) -> None:
        """Config may have 'host' and 'database' items. The config passed is expected to be:

                {'experiment_id':,
                 'emit_limit':,
                 'embed_path':}

                TODO: Automate this process for the user in builder
        """
        super().__init__(config)
        self.core = core
        self.experiment_id = self.config.get('experiment_id', str(uuid.uuid4()))
        # In the worst case, `breakdown_data` can underestimate the size of
        # data by a factor of 4: len(str(0)) == 1 but 0 is a 4-byte int.
        # Use 4 MB as the breakdown limit to stay under MongoDB's 16 MB limit.
        self.emit_limit = self.config['emit_limit']

        # create new MongoClient per OS process
        curr_pid = os.getpid()
        if curr_pid not in MongoDatabaseEmitter.client_dict:
            MongoDatabaseEmitter.client_dict[curr_pid] = MongoClient(
                config['connection_uri'])
        self.client: MongoClient = MongoDatabaseEmitter.client_dict[curr_pid]

        # extract objects from current mongo client instance
        self.db: Database = getattr(self.client, self.config.get('database', 'simulations'))
        self.history_collection: Collection = getattr(self.db, 'history')
        self.configuration: Collection = getattr(self.db, 'configuration')

        # create column indexes for the given collection objects
        self.create_indexes(self.history_collection, HISTORY_INDEXES)
        self.create_indexes(self.configuration, CONFIGURATION_INDEXES)

        self.fallback_serializer = make_fallback_serializer_function(self.core)

    def query(self, query):
        return self.history_collection.find_one(query)

    def history(self):
        return [v for v in self.history_collection.find()]

    def flush_history(self):
        for v in self.history():
            self.history_collection.delete_one(v)

    def update(self, inputs):
        self.history_collection.insert_one(inputs)
        return {}


# -- simulators -- #

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

    def __init__(self, config, core):
        """A new instance of `SmoldynProcess` based on the `config` that is passed. The schema for the config to be passed in
            this object's constructor is as follows:

            config_schema = {
                'model_filepath': 'string',  <-- analogous to python `str`
                'animate': 'bool'  <-- of type `bigraph_schema.base_types.bool`

            # TODO: It would be nice to have classes associated with this.
        """
        super().__init__(config=config, core=core)

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

        # set default starting position of molecules/particles (assume all)
        self.initial_mol_position = self.config.get('initial_mol_position', [0.0, 0.0, 0.0])
        self.initial_mol_state = self.config.get('initial_mol_state', 0)

        # get a list of the simulation species
        species_count = self.simulation.count()['species']
        counts = self.config.get('initial_species_counts')
        self.initial_species_counts = counts
        self.species_names: List[str] = []
        for index in range(species_count):
            species_name = self.simulation.getSpeciesName(index)
            if 'empty' not in species_name.lower():
                self.species_names.append(species_name)

        self.initial_species_state = {}
        self.initial_mol_state = {}
        initial_mol_counts = {spec_name: self.simulation.getMoleculeCount(spec_name, MolecState.all) for spec_name in self.species_names}
        for species_name, count in initial_mol_counts.items():
            self.initial_species_state[species_name] = count
            for _ in range(count):
                self.initial_mol_state[str(uuid4())] = {
                    'coordinates': self.initial_mol_position,
                    'species_id': species_name,
                    'state': self.initial_mol_state
                }

        # sort for logistical mapping to species names (i.e: ['a', 'b', c'] == ['0', '1', '2']
        self.species_names.sort()

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
        self.molecule_ids = list(self.initial_mol_state.keys())

        # get the simulation boundaries, which in the case of Smoldyn denote the physical boundaries
        # TODO: add a verification method to ensure that the boundaries do not change on the next step...
        self.boundaries: Dict[str, List[float]] = dict(zip(['low', 'high'], self.simulation.getBoundaries()))

        # create a re-usable counts and molecules type to be used by both inputs and outputs
        self.counts_type = {
            species_name: 'integer'
            for species_name in self.species_names
        }

        self.output_port_schema = {
            'species_counts': {
                species_name: 'integer'
                for species_name in self.species_names
            },
            'molecules': 'tree[string]',  # self.molecules_type
            'results_file': 'string'
        }

        # set time if applicable
        self.duration = self.config.get('duration')
        self.dt = self.config.get('dt', self.simulation.dt)

        # set graphics (defaults to False)
        if self.config['animate']:
            self.simulation.addGraphics('opengl_better')

        self._specs = [None for _ in self.species_names]
        self._vals = dict(zip(self.species_names, [[] for _ in self.species_names]))

    # def initial_state(self):
    #     return {
    #         'species_counts': self.initial_species_state,
    #         'molecules': self.initial_mol_state
    #     }

    def inputs(self):
        # schema = self.output_port_schema.copy()
        # schema.pop('results_file')
        # return schema
        return {}

    def outputs(self):
        return self.output_port_schema

    def update(self, inputs) -> Dict:
        # reset the molecules, distribute the mols according to self.boundarieså
        # for name in self.species_names:
        #     self.set_uniform(
        #         species_name=name,
        #         count=inputs['species_counts'][name],
        #         kill_mol=False
        #     )

        # run the simulation for a given interval if specified, otherwise use builtin time
        if self.duration is not None:
            self.simulation.run(stop=self.duration, dt=self.simulation.dt, overwrite=True)
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
            simulation_state['species_counts'][name] = counts_data[index]
            # input_counts = simulatio['species_counts'][name]
            # simulation_state['species_counts'][name] = int(final_count[index]) - input_counts

        # clear the list of known molecule ids and update the list of known molecule ids (convert to an intstring)
        # self.molecule_ids.clear()
        # for molecule in molecules_data:
            # self.molecule_ids.append(str(uuid4()))

        # get and populate the output molecules
        for i, single_mol_data in enumerate(molecules_data):
            mol_species_index = int(single_mol_data[1]) - 1
            mol_id = str(uuid4())
            simulation_state['molecules'][mol_id] = {
                'coordinates': single_mol_data[3:6],
                'species_id': self.species_names[mol_species_index],
                'state': str(int(single_mol_data[2]))
            }

        # mols = []
        # for index, mol_id in enumerate(self.molecule_ids):
        #     single_molecule_data = molecules_data[index]
        #     single_molecule_species_index = int(single_molecule_data[1]) - 1
        #     mols.append(single_molecule_species_index)
        #     simulation_state['molecules'][mol_id] = {
        #         'coordinates': single_molecule_data[3:6],
        #         'species_id': self.species_names[single_molecule_species_index],
        #         'state': str(int(single_molecule_data[2]))
        #     }

        # TODO -- post processing to get effective rates

        # TODO: adjust this for a more dynamic dir struct
        model_dir = os.path.dirname(self.model_filepath)
        for f in os.listdir(model_dir):
            if f.endswith('.txt') and 'out' in f:
                simulation_state['results_file'] = os.path.join(model_dir, f)

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


class SimulariumSmoldynStep(Step):
    """
        agent_data should have the following structure:

        {species_name(type):
            {display_type: DISPLAY_TYPE.<REQUIRED SHAPE>,
             (mass: `float` AND density: `float`) OR (radius: `float`)

    """
    config_schema = {
        'output_dest': 'string',
        'box_size': 'float',  # as per simulariumio
        'spatial_units': {
            '_default': 'nm',
            '_type': 'string'
        },
        'temporal_units': {
            '_default': 'ns',
            '_type': 'string'
        },
        'translate_output': {
            '_default': True,
            '_type': 'boolean'
        },
        'write_json': {
            '_default': True,
            '_type': 'boolean'
        },
        'run_validation': {
            '_default': True,
            '_type': 'boolean'
        },
        'file_save_name': 'maybe[string]',
        'translation_magnitude': 'maybe[float]',
        'meta_data': 'maybe[tree[string]]',
        'agent_display_parameters': 'maybe[tree[string]]'  # as per biosim simularium
    }

    def __init__(self, config, core):
        super().__init__(config=config, core=core)

        # io params
        self.output_dest = self.config['output_dest']
        self.write_json = self.config['write_json']
        self.filename = self.config.get('file_save_name')

        # display params
        self.box_size = self.config['box_size']
        self.translate_output = self.config['translate_output']
        self.translation_magnitude = self.config.get('translation_magnitude')
        self.agent_display_parameters = self.config.get('agent_display_parameters', {})

        # units params
        self.spatial_units = self.config['spatial_units']
        self.temporal_units = self.config['temporal_units']

        # info params
        self.meta_data = self.config.get('meta_data')
        self.run_validation = self.config['run_validation']

    def inputs(self):
        return {'results_file': 'string', 'species_names': 'list[string]'}

    def outputs(self):
        return {'simularium_file': 'string'}

    def update(self, inputs):
        # get job params
        in_file = inputs['results_file']
        file_data = InputFileData(in_file)

        # get species data for display data
        species_names = inputs['species_names']

        # generate simulariumio Smoldyn Data TODO: should display data be gen for each species type or n number of instances of that type?
        display_data = self._generate_display_data(species_names)
        io_data = SmoldynData(
            smoldyn_file=file_data,
            spatial_units=UnitData(self.spatial_units),
            time_units=UnitData(self.temporal_units),
            display_data=display_data,
            meta_data=self.meta_data,
            center=True
        )

        # translate reflections if needed
        if self.translate_output:
            io_data = translate_data_object(data=io_data, box_size=self.box_size, translation_magnitude=self.translation_magnitude)
        # write data to simularium file
        if self.filename is None:
            self.filename = in_file.split('/')[-1].replace('.', '') + "-simulation"

        save_path = os.path.join(self.output_dest, self.filename)
        write_simularium_file(data=io_data, simularium_fp=save_path, json=self.write_json, validate=self.run_validation)
        result = {'simularium_file': save_path + '.simularium'}

        return result

    def _generate_display_data(self, species_names) -> Dict | None:
        # user is specifying display data for agents
        if isinstance(self.agent_display_parameters, dict) and len(self.agent_display_parameters.keys()) > 0:
            display_data = {}
            for name in species_names:
                display_params = self.agent_display_parameters[name]

                # handle agent radius
                radius_param = display_params.get('radius')

                # user has passed a mass and density for a given agent
                if radius_param is None:
                    radius_param = calculate_agent_radius(m=display_params['mass'], rho=display_params['density'])

                # make kwargs for display data
                display_data_kwargs = {
                    'name': name,
                    'display_type': DISPLAY_TYPE[display_params['display_type']],
                    'radius': radius_param
                }

                # check if self.agent_params as been passed as a mapping of species_name: {species_mass: , species_shape: }
                display_data[name] = DisplayData(**display_data_kwargs)

            return display_data

        return None


# -- utils --

def generate_simularium_file(
        input_fp: str,
        dest_dir: str,
        box_size: float,
        translate_output: bool = True,
        write_json: bool = True,
        run_validation: bool = True,
        agent_parameters: Dict[str, Dict[str, Any]] = None
) -> Dict[str, str]:
    species_names = []
    float_pattern = re.compile(r'^-?\d+(\.\d+)?$')
    with open(input_fp, 'r') as f:
        output = [l.strip() for l in f.readlines()]
        for line in output:
            datum = line.split(' ')[0]
            # Check if the datum is not a float string
            if not float_pattern.match(datum):
                species_names.append(datum)
        f.close()
    species_names = list(set(species_names))

    simularium = SimulariumSmoldynStep(config={
        'output_dest': dest_dir,
        'box_size': box_size,
        'translate_output': translate_output,
        'file_save_name': None,
        'write_json': write_json,
        'run_validation': run_validation,
        'agent_display_parameters': agent_parameters,
    })

    return simularium.update(inputs={
        'results_file': input_fp,
        'species_names': species_names
    })


def make_fallback_serializer_function(process_registry) -> Callable:
    """Creates a fallback function that is called by orjson on data of
    types that are not natively supported. Define and register instances of
    :py:class:`vivarium.core.registry.Serializer()` with serialization
    routines for the types in question."""

    def default(obj: Any) -> Any:
        # Try to lookup by exclusive type
        serializer = process_registry.access(str(type(obj)))
        if not serializer:
            compatible_serializers = []
            for serializer_name in process_registry.list():
                test_serializer = process_registry.access(serializer_name)
                # Subclasses with registered serializers will be caught here
                if isinstance(obj, test_serializer.python_type):
                    compatible_serializers.append(test_serializer)
            if len(compatible_serializers) > 1:
                raise TypeError(
                    f'Multiple serializers ({compatible_serializers}) found '
                    f'for {obj} of type {type(obj)}')
            if not compatible_serializers:
                raise TypeError(
                    f'No serializer found for {obj} of type {type(obj)}')
            serializer = compatible_serializers[0]
            if not isinstance(obj, Process):
                # We don't warn for processes because since their types
                # based on their subclasses, it's not possible to avoid
                # searching through the serializers.
                warn(
                    f'Searched through serializers to find {serializer} '
                    f'for data of type {type(obj)}. This is '
                    f'inefficient.')
        return serializer.serialize(obj)
    return default


# -- Output data generators: -- #

class OutputGenerator(Step):
    config_schema = {
        'input_file': 'string',
        'context': 'string',
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        self.input_file = self.config['input_file']
        self.context = self.config.get('context')
        if self.context is None:
            raise ValueError("context (i.e., simulator name) must be specified in this processes' config.")

    @abstractmethod
    def generate(self, parameters: Optional[Dict[str, Any]] = None):
        """Abstract method for generating output data upon which to base analysis from based on its origin.

        This can be used for logic of any scope.
        NOTE: args and kwargs are not defined in this function, but rather should be defined by the
        inheriting class' constructor: i,e; start_time, etc.

        Kwargs relate only to the given simulator api you are working with.
        """
        pass

    def initial_state(self):
        # base class method
        return {
            'output_data': {}
        }

    def inputs(self):
        return {
            'parameters': 'tree[any]'
        }

    def outputs(self):
        return {
            'output_data': 'tree[any]'
        }

    def update(self, state):
        parameters = state.get('parameters') if isinstance(state, dict) else {}
        data = self.generate(parameters)
        return {'output_data': data}


class TimeCourseOutputGenerator(OutputGenerator):
    # NOTE: we include defaults here as opposed to constructor for the purpose of deliberate declaration within .json state representation.
    config_schema = {
        # 'input_file': 'string',
        # 'context': 'string',
        'start_time': {
            '_type': 'integer',
            '_default': 0
        },
        'end_time': {
            '_type': 'integer',
            '_default': 10
        },
        'num_steps': {
            '_type': 'integer',
            '_default': 100
        },
    }

    def __init__(self, config, core):
        super().__init__(config, core)
        if not self.input_file.endswith('.xml'):
            raise ValueError('Input file must be a valid SBML (XML) file')

        self.start_time = self.config.get('start_time')
        self.end_time = self.config.get('end_time')
        self.num_steps = self.config.get('num_steps')
        self.species_mapping = get_sbml_species_mapping(self.input_file)

    def initial_state(self):
        # TODO: implement this
        pass

    def generate(self, parameters: Optional[Dict[str, Any]] = None):
        # TODO: add kwargs (initial state specs) here
        executor = SBML_EXECUTORS[self.context]
        data = executor(self.input_file, self.start_time, self.end_time, self.num_steps)

        return data


