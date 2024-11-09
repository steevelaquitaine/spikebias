"""Parallel processing pipeline that stacks all chunks and simulations from scratch into a campaign

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run pipeline
    python3.9 app.py simulation --pipeline stack --parallelized True --conf 2023_01_13
    
Returns:
    _type_: Stacked spikes over all simulations of a campaign
"""

import logging
import logging.config
import os
import sys
import time

import bluepy as bp

# note: for now this module can only import
# custom packages that are in its current directory
import campaign_stacking
import chunk_stacking_parallelized as chunk_stacking_parallelized
import numpy as np
import pandas as pd
import yaml
from mpi4py import MPI

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def get_config(exp: str, simulation_date: str):
    """Choose an available experiment pipeline configuration

    Args:
        exp (str): the experiment specified to run
        - "supp/silico_reyes": in-silico simulated lfp recordings with 8 shanks, 128 contacts reyes probe
        - "buccino_2020": neuropixel probe on buccino 2020's ground truth dataset
        - "silico_neuropixels": in-silico simulated lfp recordings with neuropixel probes
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """
    logger.info("Reading experiment config.")
    if exp == "supp/silico_reyes":
        data_conf, param_conf = get_config_silico_reyes(
            simulation_date
        ).values()
    elif exp == "buccino_2020":
        data_conf, param_conf = get_config_buccino_2020(
            simulation_date
        ).values()
    elif exp == "silico_neuropixels":
        data_conf, param_conf = get_config_silico_neuropixels(
            simulation_date
        ).values()
    else:
        raise NotImplementedError(
            "The specified experiment '%s' is not implemented", exp
        )
    logger.info("Reading experiment config. - done")
    return {"dataset_conf": data_conf, "param_conf": param_conf}


def get_config_silico_reyes(simulation_date: str):
    """Get pipeline's configuration for silico Reyes probe experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/supp/silico_reyes/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "supp/silico_reyes"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/supp/silico_reyes/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_buccino_2020(simulation_date: str):
    """Get pipeline's configuration for buccino 2020 experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/buccino_2020/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "buccino_2020"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/buccino_2020/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


def get_config_silico_neuropixels(simulation_date: str):
    """Get pipeline's configuration for silico Reyes probe experiment

    Args:
        simulation_date (str): _description_

    Returns:
        dict: dataset paths and parameter configurations
    """

    with open(
        f"conf/silico_neuropixels/{simulation_date}/dataset.yml",
        "r",
        encoding="utf-8",
    ) as dataset_conf:
        dataset_conf = yaml.safe_load(dataset_conf)
        dataset_conf["exp"] = "silico_neuropixels"
        dataset_conf["date"] = simulation_date
    with open(
        f"conf/silico_neuropixels/{simulation_date}/parameters.yml",
        "r",
        encoding="utf-8",
    ) as param_conf:
        param_conf = yaml.safe_load(param_conf)
    return {"dataset_conf": dataset_conf, "param_conf": param_conf}


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


def init():
    """Get the number of parallel computing nodes

    Returns:
        _type_: _description_
    """

    # create a parallel processing object
    comm = MPI.COMM_WORLD

    # get the processes' unique identifiers
    # reminder: starts with 0
    rank = comm.Get_rank()

    # get the number of processes
    nranks = comm.Get_size()
    logger.info("%s processes were launched", nranks)
    return rank


def run(dataset_conf: dict, blue_config: dict):
    """stack all chunks into one campaign file

    Args:
        dataset_conf (dict): _description_
    """
    # launch multiple nodes
    node_rank = init()

    # run parallel processing
    chunk_stacking_parallelized.run(node_rank, dataset_conf)
    campaign_stacking.run(dataset_conf, blue_config)
    logger.info("Done")


if __name__ == "__main__":

    # start timer
    t_0 = time.time()

    # initiate log
    logger.info("Starting stacking pipeline")
    logger.info(f"The call arguments found are: {sys.argv}")

    # parse pipeline parameters
    for arg_i, argv in enumerate(sys.argv):
        logger.info(f"Argument found for {arg_i} is {argv}")
        print(argv)
        if argv == "--exp":
            EXPERIMENT = sys.argv[arg_i + 1]
        if argv == "--conf":
            CONF_DATE = sys.argv[arg_i + 1]

    # get the run config
    data_conf, param_conf = get_config(EXPERIMENT, CONF_DATE).values()

    # load campaign params
    campaign_params = load_campaign_params(data_conf)

    # run
    run(data_conf, campaign_params["blue_config"])
    logger.info(
        "Done - entire campaign was stacked in %s secs.",
        np.round(time.time() - t_0, 1),
    )
