"""Pipeline that stacks the spikes, lfp traces and cell traces of the simulations of a campaign
[TODO]: 
- fixx __main__ with blue_config
- rename simulation_stacking 
- unit-test 

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run pipeline
    python3.9 app.py simulation --pipeline stack --conf 2023_01_13

Returns:
    _type_: raw spikes over all simulations of a campaign
"""

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


def run(dataset_conf: dict, blue_config: dict):
    """stack all simulation files

    Args:
        dataset_conf (dict): _description_
    """

    # get read path
    CAMPAIGN_PATH = dataset_conf["dataeng"]["campaign"]["input"]

    # get write path
    CAMPAIGN_SPIKE_FILE = dataset_conf["dataeng"]["campaign"]["output"][
        "spike_file_path"
    ]
    CAMPAIGN_TRACE_FILE = dataset_conf["dataeng"]["campaign"]["output"][
        "trace_file_path"
    ]
    CAMPAIGN_UNIT_TRACE_FILE = dataset_conf["dataeng"]["campaign"]["output"][
        "unit_trace_path"
    ]

    # Trace and spikes are in the same folder
    CAMPAIGN_FILE_PATH = os.path.dirname(CAMPAIGN_SPIKE_FILE)

    # get timestep
    LFP_DT = float(blue_config.Report_lfp["Dt"])
    SPIKE_DT = float(blue_config["Run_Default"]["Dt"])

    # get sorted simulation identifiers
    simulation_id = os.listdir(CAMPAIGN_PATH)
    simulation_id = list(map(int, simulation_id))
    simulation_id = sorted(simulation_id)
    n_sim = len(simulation_id)

    # initialize
    spike_list = []
    trace_list = []
    spike_current_duration = 0
    lfp_current_duration = 0

    # log
    logger.info("Started stacking simulations for lfp traces and spikes ...")

    # stack simulation spike into campaign level
    for sim_i in range(n_sim):

        # get file path of simulation's raw spikes and traces
        spike_file = os.path.join(
            dataset_conf["dataeng"]["simulations"]["input"],
            str(simulation_id[sim_i]) + "/stacked/spiketrains.pkl",
        )
        trace_file = os.path.join(
            dataset_conf["dataeng"]["simulations"]["input"],
            str(simulation_id[sim_i]) + "/stacked/traces.pkl",
        )

        # get list of near-probe cell gids processed
        near_probe_cells = pd.read_pickle(
            f"""{dataset_conf["dataeng"]["simulations"]["input"]}{simulation_id[sim_i]}/chunks/cells/info_near_probe_cells.pkl"""
        )

        # read spikes and traces
        # WARNING! each unit trace .pkl file that contains the unsummed traces
        # for each unit for one simulation sizes 57GB which produces an out of
        # memory error. So a trace file was saved for each single unit (1.4MB)
        spike = pd.read_pickle(spike_file)
        trace = pd.read_pickle(trace_file)

        # track simulation duration and
        # increment all spike and trace times
        spike.index += spike_current_duration
        trace.index += lfp_current_duration

        # append them
        spike_list.append(spike)
        trace_list.append(trace)

        # track campaign duration
        spike_current_duration = trace.index[-1] + SPIKE_DT
        lfp_current_duration = trace.index[-1] + LFP_DT

    # stack spike and lfp trace simulations
    campaign_spike = pd.concat(spike_list)
    campaign_trace = pd.concat(trace_list)

    # create path and write
    if not os.path.isdir(CAMPAIGN_FILE_PATH):
        os.makedirs(CAMPAIGN_FILE_PATH)
    campaign_spike.to_pickle(CAMPAIGN_SPIKE_FILE)
    campaign_trace.to_pickle(CAMPAIGN_TRACE_FILE)

    # log
    logger.info(
        "Done - Spike and lfp simulations were written into one campaign in %s",
        CAMPAIGN_FILE_PATH,
    )

    # log
    logger.info(
        "Started stacking cell traces over simulations into one campaign ..."
    )

    # write each cell trace raw over simulations
    for cell_i in near_probe_cells.values:

        # initialize
        cell_trace_list = []

        # reset current duration
        current_duration = 0

        # stack simulations
        for sim_i in range(n_sim):

            # read cell trace
            cell_trace_file = os.path.join(
                f"""{dataset_conf["dataeng"]["simulations"]["input"]}{simulation_id[sim_i]}/stacked/cell_{cell_i[0]}_trace.pkl"""
            )
            cell_trace = pd.read_pickle(cell_trace_file)

            # track simulation duration and
            # increment all unit trace times
            cell_trace.index += current_duration

            # append them
            cell_trace_list.append(cell_trace)

            # track campaign duration
            current_duration = cell_trace.index[-1] + LFP_DT

        # stack cell simulations
        campaign_unit_trace = pd.concat(cell_trace_list)

        # create path and write
        if not os.path.isdir(CAMPAIGN_UNIT_TRACE_FILE):
            os.makedirs(CAMPAIGN_UNIT_TRACE_FILE)
        campaign_unit_trace.to_pickle(
            f"{CAMPAIGN_UNIT_TRACE_FILE}cell_{cell_i[0]}_trace.pkl"
        )

    # copy list of cells to raw cells path
    near_probe_cells.to_pickle(
        f"{CAMPAIGN_UNIT_TRACE_FILE}info_near_probe_cells.pkl"
    )

    # log
    logger.info(
        "Done - Cell trace simulations were written into one campaign in %s",
        CAMPAIGN_UNIT_TRACE_FILE,
    )
    return {
        "spike": campaign_spike,
        "trace": campaign_trace,
    }
