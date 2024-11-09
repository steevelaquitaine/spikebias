import logging
import logging.config
import os
import sys

import bluepy as bp
import numpy as np
import pandas as pd
import probeinterface as pi
import spikeinterface.extractors as se
import yaml
from sklearn.decomposition import PCA

from src.nodes.dataeng.silico.filtering import filter_microcircuit_cells
from src.nodes.utils import create_if_not_exists
from src.pipes.silico.dataeng.loading import load_campaign_params

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def load_sim_piece(i: int, simulation_results: str):
    """Load a chunk of one simulation

    Args:
        i (int): _description_
        simulation_results (str): _description_

    Returns:
        _type_: _description_
    """

    # get write path
    WRITE_PATH = os.path.dirname(simulation_results)

    # create path
    create_if_not_exists(WRITE_PATH)

    # write simulation results
    traces_piece = pd.read_pickle(
        simulation_results + "traces" + str(i) + ".pkl"
    )
    return traces_piece


def shift_shank(original_probe, i, d):
    pr = original_probe.copy()
    pr.move([i * d, 0])
    return pr


def create_reyes_probe(Y_PITCH, X_PITCH, N_SHANK, N_ELEC, R_ELEC):
    """create Reyes 16x8's multi-shank recording probe

    Args:
        Y_PITCH (_type_): _description_
        X_PITCH (_type_): _description_
        N_SHANK (_type_): _description_
        N_ELEC (_type_): _description_
        R_ELEC (_type_): _description_

    Returns:
        _type_: _description_
    """
    shank = pi.generator.generate_linear_probe(
        num_elec=N_ELEC,
        ypitch=Y_PITCH,
        contact_shapes="circle",
        contact_shape_params={"radius": R_ELEC},
    )
    shank.create_auto_shape(probe_type="tip", margin=50)
    shanks = [shift_shank(shank, i, X_PITCH) for i in range(N_SHANK)]
    probe = pi.combine_probes(shanks)
    multi_shank = probe.to_3d()
    multi_shank.rotate(90, axis=[0, 0, 1])
    multi_shank.move(
        np.array([-(X_PITCH * N_SHANK) / 2, 0, -(Y_PITCH * N_SHANK) / 2])
    )
    return multi_shank


def calculate_probe_motion_params(somaPos):
    center = np.mean(somaPos, axis=0)

    pca = PCA(n_components=3)
    pca.fit(somaPos)
    main_axis = pca.components_[0]

    elevation = np.arctan2(
        np.sqrt(main_axis[0] ** 2 + main_axis[1] ** 2), main_axis[2]
    )
    azimuth = np.arctan2(main_axis[1], main_axis[0])
    return center, elevation, azimuth


def set_probe_in_place(n_sites, multi_shank, center, elevation, azimuth):
    multi_shank.rotate(elevation * 180 / np.pi, axis=[0, 1, 0])
    multi_shank.rotate(azimuth * 180 / np.pi, axis=[0, 0, 1])
    multi_shank.move(center)
    multi_shank.set_device_channel_indices(np.arange(n_sites))


def position_probe(n_sites, somaPos, multi_shank):
    center, elevation, azimuth = calculate_probe_motion_params(somaPos)

    # set the probe in place for recording
    set_probe_in_place(n_sites, multi_shank, center, elevation, azimuth)


def get_recording_trace(dataset_conf: dict, param_conf: dict):

    # set probe parameters
    Y_PITCH = param_conf["probe"]["reyes_16_x_8"]["y_pitch"]
    X_PITCH = param_conf["probe"]["reyes_16_x_8"]["x_pitch"]
    N_SHANK = param_conf["probe"]["reyes_16_x_8"]["n_shank"]
    N_ELEC = param_conf["probe"]["reyes_16_x_8"]["n_elec"]
    R_ELEC = param_conf["probe"]["reyes_16_x_8"]["r_elec"]

    # set read path
    TRACE_FILE_PATH = dataset_conf["dataeng"]["campaign"]["output"][
        "trace_file_path"
    ]

    # get campaign parameters from one simulation
    simulation = load_campaign_params(dataset_conf, param_conf)

    # get the soma positions of the microcircuit neurons
    # to position probe
    filtered_cells = filter_microcircuit_cells(simulation)

    # create the probe
    probe = create_reyes_probe(Y_PITCH, X_PITCH, N_SHANK, N_ELEC, R_ELEC)

    # position the probe for recording
    position_probe(
        simulation["n_sites"], filtered_cells["soma_location"], probe
    )

    # read trace
    trace = pd.read_pickle(TRACE_FILE_PATH)

    # format as a spikeinterface recording
    recording = se.NumpyRecording(
        traces_list=[np.array(trace)],
        sampling_frequency=simulation["lfp_sampling_freq"],
    )

    # map traces to probe channels
    return recording.set_probe(probe)


def run(dataset_conf: dict, param_conf: dict):
    return get_recording_trace(dataset_conf, param_conf)


if __name__ == "__main__":
    """Entry point"""
    # parse pipeline parameters
    conf_date = sys.argv[1]

    # read the pipeline's configuration
    with open(f"conf/silico/{conf_date}/dataset.yml") as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)

    with open(f"conf/silico/{conf_date}/parameters.yml") as param_conf:
        param_conf = yaml.safe_load(param_conf)

    # run
    output = run(dataset_conf, param_conf)
