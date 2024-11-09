"""Loading pipeline

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run
    python3.9 app.py simulation --pipeline load_raw --conf 2023_01_13

Returns:
    _type_: _description_
"""
import logging
import logging.config
import os
import sys
from sys import argv
import spikeinterface as si
import bluepy as bp
import h5py
import numpy as np
import pandas as pd
import yaml

from src.nodes.utils import create_if_not_exists, get_config

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def load_prep_recording(data_conf: dict):
    return si.load_extractor(
        data_conf["preprocessing"]["output"]["trace_file_path"]
    )


def load_campaign_params(data_conf: dict):
    """Load the configuration of one simulation of the campaign produced
    by one BlueConfig file

    TODO:
    - simplify:
        - ideally reduce all to "blue_config = bp.Simulation("BlueConfig").config" ...
        - ... and calculate sampling frequencies from config parameters
    - move to src.nodes

    Returns:
        dict: the campaign parameters
    """
    # read simulation
    simulation = bp.Simulation(data_conf["dataeng"]["blueconfig"])
    
    # get circuit
    circuit = simulation.circuit

    # get lfp report
    report = simulation.report("lfp")

    # get number of neurons
    n_neurons = len(report.gids)

    # get data
    _, _, N_SITES = get_trace(report, n_neurons)

    # calculate the sampling frequency in Hz from report time step
    # (reporting Dt in ms, typically 0.05 ms, under "report lfp" section)
    LFP_SAMPLING_FREQ = (1 / report.t_step) * 1000

    # calculate the spike sampling frequency in Hz
    # (Simulation's Dt in ms typically 0.025 ms associated
    # with CoreNeuron simulation parameters under "run Default"
    # section)
    SPIKE_SAMPLING_FREQ = (
        1 / float(simulation.config["Run_Default"]["Dt"])
    ) * 1000

    # this should be the same for all pieces
    # locations = load_locations_from_weights(weightspath, N_SITES)

    # recursively create writing path, if not exists
    # create_if_not_exists(CHANNEL_LOCATION_PATH)

    # write channel locations
    # np.save(CHANNEL_LOCATION_FILE, locations, allow_pickle=True)

    # log
    # logger.info(
    #     "Channel locations have been written in file: %s",
    #     CHANNEL_LOCATION_FILE,
    # )
    return {
        "blue_config": simulation.config,
        "paths": {
            "BlueConfig_path": data_conf["dataeng"]["blueconfig"],
        },
        "lfp_sampling_freq": LFP_SAMPLING_FREQ,
        "spike_sampling_freq": SPIKE_SAMPLING_FREQ,
        "n_sites": N_SITES,
        "circuit": circuit,
    }


def get_trace(report, n_neurons: int):
    """Get lfp traces

    TODO:
    - move to src.nodes

    Args:
        report (_type_): _description_
        n_neurons (int): _description_

    Returns:
        _type_: _description_
    """
    data = report.get(
        t_start=report.meta["start_time"],
        t_end=report.meta["start_time"] + report.meta["time_step"],
    )

    # get neurons ids
    gids_id = pd.unique(data.columns)

    # get number of recording sites
    N_SITES = int(data.shape[1] / n_neurons)
    data.columns = pd.MultiIndex.from_product(
        [gids_id, np.arange(N_SITES)], names=["gid", "contact"]
    )

    # get channel traces by summing over neurons' voltage
    trace = data.groupby(level="contact", axis=1).sum() * 1000
    return trace, data, N_SITES


def read_simulation(data_conf: dict):
    """Load a simulation with BluePy from its BlueConfig

    TODO:
    - move to src.nodes

    Args:
        data_conf (dict): _description_

    Returns:
        _type_: _description_
    """
    simulation = bp.Simulation(data_conf["dataeng"]["blueconfig"])
    single_piece_sim = True
    BlueConfig_path = data_conf["dataeng"]["blueconfig"]
    return simulation, single_piece_sim, BlueConfig_path


if __name__ == "__main__":

    # parse pipeline parameters
    CONF_DATE = sys.argv[1]
    EXPERIMENT = None
    for arg_i, argument in enumerate(argv):
        if argument == "--exp":
            EXPERIMENT = argv[arg_i + 1]

    # get experiment config
    data_conf, param_conf = get_config(EXPERIMENT, CONF_DATE).values()

    # load simulation params
    simulation_params = load_campaign_params(data_conf)
