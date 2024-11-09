"""Chunking of a single simulation
note: It takes 1 hour for a 4 sec simulation

author: steeve.laquitaine@epfl.ch

usage :
    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run pipeline
    python3.9 app.py simulation --exp supp/silico_reyes --pipeline chunk --conf 2023_01_13
"""

import logging
import logging.config
import os
import sys
from time import time

import bluepy as bp
import h5py
import numpy as np
import pandas as pd
import yaml

from src.nodes.utils import get_config

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def load_locations_from_weights(data_conf: dict, nsites: int):
    """load channel locations from the channel weight file
    of this experiment

    Args:
        data_conf (dict): _description_
        nsites (int): _description_

    Returns:
        _type_: _description_
    """

    # get weight file path
    WEIGHT_PATH = data_conf["dataeng"]["chunks"]["input"]["weight"]

    # select experiment config
    if data_conf["exp"] == "supp/silico_reyes":
        logger.info("loading %s experiment config", data_conf["exp"])
        return load_locations_from_weights_reyes(WEIGHT_PATH, nsites)
    elif data_conf["exp"] == "silico_neuropixels":
        logger.info("loading %s experiment config", data_conf["exp"])
        return load_locations_from_weights_neuropixels(WEIGHT_PATH, nsites)


def load_locations_from_weights_reyes(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        _type_: _description_
    """
    # loading weights file with channel locations
    f = h5py.File(weightspath, "r")

    locationstmp = np.array([0, 0, 0])
    for i in range(nsites):
        temp = np.array(f["electrodes"]["reyespuerta_" + str(i)]["location"])
        locationstmp = np.c_[locationstmp, temp]
    locations = locationstmp.T[1:]
    return locations


def load_locations_from_weights_neuropixels(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        _type_: _description_
    """
    raise NotImplementedError()


def find_close_gids(cells, probepos, radius: int):
    cpos = cells[list("xyz")]
    tiled = np.tile(probepos, (cpos.shape[0], 1))
    dist = np.linalg.norm(cpos - tiled, axis=1)  # TODO: Is it euclidean?
    return cells[dist < radius]


def run(data_conf: dict, param_conf: dict):
    # start timer
    t_0 = time()

    # get write paths
    OUT_PATH = data_conf["dataeng"]["chunks"]["output"]["chunks"]
    EXPERIMENT = os.path.basename(
        data_conf["dataeng"]["chunks"]["input"]["data"]
    )
    CAMPAIGN_PATH = data_conf["dataeng"]["chunks"]["input"]["data"]

    # get parameters
    N_CHUNKS = param_conf["circuit"]["n_trace_spike_files"]
    RADIUS = param_conf["circuit"]["radius"]

    # get sorted simulation identifiers
    simulation_id = os.listdir(CAMPAIGN_PATH)
    simulation_id = list(map(int, simulation_id))
    simulation_id = sorted(simulation_id)
    n_sim = len(simulation_id)

    # log
    logger.info(f"Chunking simulation campaign: {CAMPAIGN_PATH}")
    logger.info(f"{n_sim} simulations were found: {simulation_id}")

    # get each simulation
    for sim_i in simulation_id:
        # log
        logger.info(f"Processing simulation id: {sim_i}")

        # chunk it, get the chunk's data and save them
        for chunk_i in np.arange(0, N_CHUNKS):
            # setup read paths
            simpath = CAMPAIGN_PATH + "/" + str(sim_i)

            # setup write paths
            out = OUT_PATH + "/" + EXPERIMENT

            # read simulation
            sim = bp.Simulation(simpath + "/BlueConfig")

            # get report (takes a few secs)
            t_0_rep = time()
            report = sim.report("lfp")

            # log
            logger.info(
                f"Loading report took {np.round(time() - t_0_rep,1)} secs."
            )

            # calculate the start and end time of this chunk
            duration = report.meta["end_time"]
            start_time = duration / N_CHUNKS * chunk_i
            end_time = duration / N_CHUNKS * (chunk_i + 1)

            # collect this chunk's trace data (takes a few secs)
            # voltage against gid, channel, time
            try:
                data = report.get(t_start=start_time, t_end=end_time)
            except:
                # for the last chunk
                data = report.get(t_start=start_time)

            gids_id = pd.unique(data.columns)
            nsites = int(data.shape[1] / len(report.gids))
            data.columns = pd.MultiIndex.from_product(
                [gids_id, np.arange(nsites)], names=["gid", "contact"]
            )

            # sum data over cell to create trace for each channel contact
            traces_piece = data.groupby(level="contact", axis=1).sum() * 1000

            # log
            logger.info(
                f"Done with trace chunk ending at {traces_piece.index[-1]}"
            )

            # load channel locations
            locations = load_locations_from_weights(data_conf, nsites)

            # find near-probe cells within RADIUS microns to one contact
            cells = sim.circuit.cells.get({"$target": "hex0"})
            all_close_gids = []
            for location in locations:
                close_gids_df = find_close_gids(cells, location, RADIUS)
                close_gids = list(close_gids_df.index)
                all_close_gids = all_close_gids + close_gids
            all_close_gids = list(set(all_close_gids))

            # get the individual traces of near-probe cells
            near_probe_individual_unit_trace = data[all_close_gids]

            # get matching spikes from near-probe cells
            spikes_piece = sim.spikes.get(
                gids=all_close_gids, t_start=start_time, t_end=end_time
            )

            # log
            logger.info(
                f"Done with spike chunk ending at {spikes_piece.index[-1]}"
            )

            # check path exists for writing spike and trace
            if not os.path.isdir(
                OUT_PATH + "/simulations/" + str(sim_i) + "/chunks/cells/"
            ):
                os.makedirs(
                    OUT_PATH + "/simulations/" + str(sim_i) + "/chunks/cells"
                )

            # save channel locations
            if not os.path.exists(OUT_PATH + "/"):
                os.makedirs(OUT_PATH + "/")

            # write each near-probe cell trace only (1.4MB)
            for gid in all_close_gids:
                near_probe_individual_unit_trace[gid].to_pickle(
                    f"{OUT_PATH}/simulations/{sim_i}/chunks/cells/cell_{gid}_chunk_{chunk_i}_trace.pkl"
                )

            # write lfp traces
            traces_piece.to_pickle(
                OUT_PATH
                + "/simulations/"
                + str(sim_i)
                + "/chunks/traces"
                + str(chunk_i)
                + ".pkl"
            )

            # write near-probe spikes
            spikes_piece.to_pickle(
                OUT_PATH
                + "/simulations/"
                + str(sim_i)
                + "/chunks/spikes"
                + str(chunk_i)
                + ".pkl"
            )

            # write list of near-probe cells
            pd.DataFrame(all_close_gids).to_pickle(
                f"{OUT_PATH}/simulations/{sim_i}/chunks/cells/info_near_probe_cells.pkl"
            )

            # write channel locations
            np.save(out + "_locations", locations, allow_pickle=True)

            # calculate time
            duration = np.round(time() - t_0, 1)

            # log
            logger.info(
                f"Wrote spike and trace files for chunk {chunk_i} (starts at {start_time}, ends at {end_time}) of {N_CHUNKS} for simulation {sim_i} of {n_sim} in {duration} secs"
            )
            logger.info(
                "Campaign was chunked in  %s secs", round(time() - t_0, 1)
            )


if __name__ == "__main__":
    # initiate log
    logger.info("Starting chunking pipeline")
    logger.info(f"The call arguments found are: {sys.argv}")

    # parse pipeline parameters
    for arg_i, argv in enumerate(sys.argv):
        logger.info(f"Recorded arguments for is {arg_i} : {argv}")
        print(argv)
        if argv == "--exp":
            EXPERIMENT = argv[arg_i + 1]
        if argv == "--conf":
            conf_date = sys.argv[arg_i + 1]

    # get run config
    data_conf, param_conf = get_config(EXPERIMENT, conf_date).values()

    # run
    run(data_conf, param_conf)
