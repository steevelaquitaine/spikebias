## Validation of EM assumptions

Here, we validate the assumption that using a point electrode instead of a finite electrode has limited impact on the results. We know that the potential recorded by a spherical electrode in an infinite homogeneous medium is identical to that recorded by a point electrode, for sources outside of the sphere.

We compare the weights produced for infinitesimal electrodes to weights calculated under the assumption that the minimum distance between a neural segment and an electrode is 6 um (given that a Neuropixels contact is 12x12 um). 

To generate the weights files, install [the latest version of BlueRecording](https://github.com/joseph-tharayil/BlueRecording/tree/master) according to the instructions in the ReadMe. Then download the circuit model, and calculate the positions of the neural segments in the model, as described in the BlueRecording ReadMe. Alternatively, the positions are available in [this Zenodo repository](https://zenodo.org/records/14998743). Then, run the script `run_initialize_sphereSizes.py` to initialize the weights files. Then, run the script `run_write_weights_sphereSizes.py` to populate the weights files. We recommend that you run the latter scripy using MPI, with `mpirun -n <your number of cores> python run_write_weights_sphereSizes.py`

