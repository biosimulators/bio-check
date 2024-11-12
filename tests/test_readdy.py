import os
import numpy as np
import readdy
system = readdy.ReactionDiffusionSystem(box_size=[10., 10., 10.])
system.add_species("A", 1.0)
sim = system.simulation("SingleCPU")
sim.output_file = "out.h5"
init_pos = np.random.uniform(size=(100, 3)) * 10. - 5.
sim.load_particles_from_latest_checkpoint("checkpoints/") if os.path.exists("checkpoints/") else sim.add_particles("A", init_pos)
sim.make_checkpoints(100, output_directory="checkpoints/", max_n_saves=5)
os.remove(sim.output_file) if os.path.exists(sim.output_file) else None
sim.run(1000, 0.1)
