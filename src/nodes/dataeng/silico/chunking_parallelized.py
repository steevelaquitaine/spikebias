"""Chunking of each single simulation to aggregate lfp trace over cell traces

       author: steeve.laquitaine@epfl.ch
modified from: Milo Imbeni

usage :

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # setup "sbatch/parallelize_chunking.sbatch" and run pipeline
    sbatch sbatch/parallelize_chunking.sbatch

Stats:
    - takes 1 hour to process a 30 secs simulation on 10 nodes
"""

import logging
import logging.config
import os
import sys
import time

import bluepy as bp
import h5py
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


def init():
    """_summary_

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
    logger.info(f"(init) {nranks} processes were launched")
    return rank


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
        logger.info(
            "loading %s experiment %s date config",
            data_conf["exp"],
            data_conf["date"],
        )
        return load_locations_from_weights_reyes(WEIGHT_PATH, nsites)
    elif data_conf["exp"] == "silico_neuropixels":
        logger.info(
            "loading %s experiment %s date config",
            data_conf["exp"],
            data_conf["date"],
        )
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
    f.close()
    return locations


def load_locations_from_weights_neuropixels(weightspath: str, nsites: int):
    """Load the channel site locations from the neuropixel weights file

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        _type_: _description_
    """
    # loading weights file with channel locations
    f = h5py.File(weightspath, "r")

    # get the list of contact ids in the file
    contact_ids = list(f["electrodes"].keys())

    # load the contact locations
    locationstmp = np.array([0, 0, 0])
    for site_i in range(nsites):
        temp = np.array(f["electrodes"][contact_ids[site_i]]["location"])
        locationstmp = np.c_[locationstmp, temp]
    locations = locationstmp.T[1:]
    f.close()
    return locations


def find_close_gids(cells, probepos, radius: int):
    """_summary_

    Args:
        cells (_type_): _description_
        probepos (_type_): _description_
        radius (int): _description_

    Returns:
        _type_: _description_
    """
    cpos = cells[list("xyz")]
    tiled = np.tile(probepos, (cpos.shape[0], 1))
    dist = np.linalg.norm(cpos - tiled, axis=1)
    return cells[dist < radius]


def run(node_rank: int, data_conf: dict, param_conf: dict):
    """run the chunking pipeline
    TODO:
    - refactor: modularize the common code with the "chunking" module

    Args:
        node_rank (int): rank assigned to the cluster node, read with MPI
        data_conf (dict): config paths
        - note: simulation ids should start with id 1
        param_conf (dict): _description_
    """

    # get write paths
    OUT_PATH = data_conf["dataeng"]["chunks"]["output"]["chunks"]
    EXPERIMENT = data_conf["campaign"]["exp"]
    RUN = data_conf["campaign"]["run"]
    CAMPAIGN_PATH = data_conf["dataeng"]["chunks"]["input"]["data"]

    # get parameters
    N_CHUNKS = param_conf["circuit"]["n_trace_spike_files"]
    RADIUS = param_conf["circuit"]["radius"]

    # log
    logger.info("Chunking simulation campaign %s: ", CAMPAIGN_PATH)

    # init timer
    t_0 = time.time()

    # set the simuation id to process in this node
    simulation_ids_in_campaign = []
    for folder in os.listdir(CAMPAIGN_PATH):
        if folder.isnumeric():
            simulation_ids_in_campaign.append(folder)
    logger.info(f"Chunking simulation campaign: {simulation_ids_in_campaign}")
    simulation_id = simulation_ids_in_campaign[node_rank]

    # log
    logger.info(f"Processing simulation id: {simulation_id}")

    # Chunk it, get the chunk's data and save them
    for chunk_i in np.arange(0, N_CHUNKS):

        # report
        logger.info("Processing chunk %s:", chunk_i)

        # setup read paths
        simpath = os.path.join(CAMPAIGN_PATH, simulation_id)

        # setup write paths
        out = os.path.join(OUT_PATH, EXPERIMENT, RUN)

        # read simulation
        sim = bp.Simulation(os.path.join(simpath, "BlueConfig"))

        # get report (takes a few secs)
        t_0_rep = time.time()
        report = sim.report("lfp")

        # log
        logger.info(
            "Loading report - done - took %s secs",
            np.round(time.time() - t_0_rep, 1),
        )

        # calculate the start and end time of this chunk
        duration = report.meta["end_time"]
        start_time = duration / N_CHUNKS * chunk_i
        end_time = duration / N_CHUNKS * (chunk_i + 1)

        # collect this chunk's data
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

        # sum data over compartments for each channel contact
        traces_piece = data.groupby(level="contact", axis=1).sum() * 1000

        # log
        logger.info(
            "Done with trace chunk ending at %s", traces_piece.index[-1]
        )

        # load channel locations
        locations = load_locations_from_weights(data_conf, nsites)

        # find all cells closer than radius to at least one contact
        cells = sim.circuit.cells.get({"$target": "hex0"})
        all_close_gids = []
        for location in locations:
            close_gids_df = find_close_gids(cells, location, RADIUS)
            close_gids = list(close_gids_df.index)
            all_close_gids = all_close_gids + close_gids
        all_close_gids = list(set(all_close_gids))

        # get the individual traces of near-probe cells
        near_probe_individual_unit_trace = data[all_close_gids]

        # get matching spikes
        spikes_piece = sim.spikes.get(
            gids=all_close_gids, t_start=start_time, t_end=end_time
        )

        # log
        logger.info(
            "Done with spike chunk ending at %s", spikes_piece.index[-1]
        )

        # write traces and spikes
        if not os.path.isdir(
            os.path.join(out, "simulations", simulation_id, "chunks/cells")
        ):
            os.makedirs(
                os.path.join(out, "simulations", simulation_id, "chunks/cells")
            )

        # save channel locations
        if not os.path.exists(out + "/"):
            os.makedirs(out + "/")

        # write results
        # store near-probe individual unit traces only (up to 2GB for 130 ms)
        # write each near-probe cell trace only (1.4MB)
        for gid in all_close_gids:
            near_probe_individual_unit_trace[gid].to_pickle(
                os.path.join(
                    out,
                    "simulations",
                    simulation_id,
                    f"chunks/cells/cell_{gid}_chunk_{chunk_i}_trace.pkl",
                )
            )

        # store lfp traces
        traces_piece.to_pickle(
            os.path.join(
                out,
                "simulations",
                simulation_id,
                f"chunks/traces{chunk_i}.pkl",
            )
        )
        # store near-probe spikes
        spikes_piece.to_pickle(
            os.path.join(
                out,
                "simulations",
                simulation_id,
                f"chunks/spikes{chunk_i}.pkl",
            )
        )

        # write list of near-probe cells
        pd.DataFrame(all_close_gids).to_pickle(
            os.path.join(
                out,
                "simulations",
                simulation_id,
                "chunks/cells/info_near_probe_cells.pkl",
            )
        )

        # write channel locations
        np.save(
            os.path.join(
                out, "channel_locations", locations, allow_pickle=True
            )
        )

        # log
        logger.info(
            f"Done with chunk {chunk_i} of {N_CHUNKS} for simulation {simulation_id} in {np.round(time.time() - t_0, 1)} secs"
        )


if __name__ == "__main__":

    # initiate log
    logger.info("Starting chunking pipeline")
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

    # launch parallel processing
    simulation_id = init()

    # process simulation data
    run(simulation_id, data_conf, param_conf)
