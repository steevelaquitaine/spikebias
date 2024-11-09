"""Pipeline that stacks all chunks of all simulations of a campaign

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run pipeline
    python3.9 app.py simulation --pipeline stacking.py --conf 2023_01_13

Returns:
    _type_: Stacks all chunks of all a simulation campaign
"""

import glob
import logging
import logging.config
import os
import sys
import time

import numpy as np
import pandas as pd
import yaml

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def stack_chunks(simulation_id: int, dataset_conf: dict):
    """stack the chunks of a simulation

    Args:

    Returns:
        _type_: _description_
    """

    # get path of the simulation chunks
    CHUNK_PATH = os.path.join(
        dataset_conf["dataeng"]["simulations"]["input"],
        str(simulation_id) + "/chunks/",
    )

    # get write path of stacked simulations
    SPIKE_STACKED_FILENAME_PATH = os.path.join(
        dataset_conf["dataeng"]["simulations"]["input"],
        str(simulation_id) + "/stacked/spiketrains.pkl",
    )
    TRACE_STACKED_FILENAME_PATH = os.path.join(
        dataset_conf["dataeng"]["simulations"]["input"],
        str(simulation_id) + "/stacked/traces.pkl",
    )
    UNIT_TRACE_STACKED_PATH = os.path.join(
        dataset_conf["dataeng"]["simulations"]["input"],
        str(simulation_id) + "/stacked/",
    )

    # both spike and trace are in the same folder
    STACKED_PATH = os.path.dirname(SPIKE_STACKED_FILENAME_PATH)

    # log
    logger.info("Reading from simulation path: %s", CHUNK_PATH)

    # count chunks (same number of spike file than trace)
    n_chunks = len(glob.glob(os.path.join(CHUNK_PATH, "spike*")))

    # write each unit trace
    near_probe_cells = pd.read_pickle(
        f"{CHUNK_PATH}cells/info_near_probe_cells.pkl"
    )

    # initialize
    simulation_spike = []
    simulation_trace = []

    # append chunks
    for chunk_i in range(0, n_chunks):

        # read chunk
        chunk_spike = pd.read_pickle(
            CHUNK_PATH + "spikes" + str(chunk_i) + ".pkl"
        )
        chunk_trace = pd.read_pickle(
            CHUNK_PATH + "traces" + str(chunk_i) + ".pkl"
        )

        # check if single chunk
        if n_chunks == 1:
            if chunk_i == 0:
                simulation_spike.append(chunk_spike)
                simulation_trace.append(chunk_trace)
        else:
            simulation_spike.append(chunk_spike)
            simulation_trace.append(chunk_trace)

    # stack chunks
    stacked_spike = pd.concat(simulation_spike)
    stacked_trace = pd.concat(simulation_trace)

    # create path and write
    if not os.path.isdir(STACKED_PATH):
        os.makedirs(STACKED_PATH)
    stacked_spike.to_pickle(SPIKE_STACKED_FILENAME_PATH)
    stacked_trace.to_pickle(TRACE_STACKED_FILENAME_PATH)

    # stack by cell and write
    for cell_i in near_probe_cells.values:
        simulation_trace = []
        for chunk_i in range(0, n_chunks):
            chunk_trace = pd.read_pickle(
                f"{CHUNK_PATH}cells/cell_{cell_i[0]}_chunk_{chunk_i}_trace.pkl"
            )
            simulation_trace.append(chunk_trace)
        stacked_unit_trace = pd.concat(simulation_trace)
        stacked_unit_trace.to_pickle(
            f"{UNIT_TRACE_STACKED_PATH}cell_{cell_i[0]}_trace.pkl"
        )

    # log
    logger.info(
        "Stacked spike, lfp trace, unit trace files were written in: %s",
        STACKED_PATH,
    )
    return {
        "spike": stacked_spike,
        "trace": stacked_trace,
    }


def run(dataset_conf: dict):
    """_summary_

    Args:
        dataset_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # get campaign path
    CAMPAIGN_PATH = dataset_conf["dataeng"]["simulations"]["input"]

    # get sorted simulation identifiers
    simulation_id = os.listdir(CAMPAIGN_PATH)
    simulation_id = list(map(int, simulation_id))
    simulation_id = sorted(simulation_id)
    n_sim = len(simulation_id)

    # stack spikes for each simulation
    for sim_i in range(n_sim):
        output = stack_chunks(
            simulation_id[sim_i],
            dataset_conf,
        )
    return output
