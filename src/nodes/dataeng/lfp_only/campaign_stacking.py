"""Stacks simulation traces and spikes into single campaign files

author: steeve.laquitaine@epfl.ch 

Returns:
    _type_: raw spikes and traces stacked over simulations
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

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def get_simulation_ids(campaign_path: str):
    """_summary_

    Args:
        campaign_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get sorted simulation identifiers
    simulation_id = os.listdir(campaign_path)

    # exclude bbp_worflow irrelevant output files
    simulation_id.remove("config.json")
    simulation_id = [
        val for val in simulation_id if not val.endswith(".script")
    ]

    # cast to integers
    simulation_id = list(map(int, simulation_id))

    # sort
    simulation_id = sorted(simulation_id)
    return simulation_id


def load_locations_from_weights(data_conf: dict, nsites: int):
    """load channel 3D coordinates from the channel weight file
    of this experiment, for the tested runs (dates)

    Args:
        data_conf (dict): _description_
        nsites (int): _description_

    Returns:
        np.array: weight 3D coordinates
    """

    # select experiment config
    # case npx 10min 384ch hex_O    
    if data_conf["exp"] == "supp/silico_reyes":
        # get weight path 
        WEIGHT_PATH = data_conf["dataeng"]["chunks"]["input"]["weight"]
        return load_locations_from_weights_reyes(WEIGHT_PATH, nsites)
    
    # case neuropixel probes
    elif data_conf["exp"] == "silico_neuropixels":
        # case npx 10min 384ch hex_O
        if data_conf["date"] == "npx_spont/sims/2023_06_26":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )
        
        # case npx 10min 384ch hex_O1        
        if data_conf["date"] == "npx_spont/sims/2023_08_17":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch_hex_O1(
                WEIGHT_PATH, nsites - 1
            )
        # case npx 10min 384ch hex_O rou:0.3 pfr:0.5    
        # the probe sites are located with respect to hex_01
        # not hex_0
        if data_conf["date"] == "npx_spont/sims/2023_09_12":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch_hex_O1(
                WEIGHT_PATH, nsites - 1
            )        
        # case npx 10min 384ch hex_O rou:0.3 pfr:0.9    
        # the probe sites are located with respect to hex_01
        # not hex_0
        if data_conf["date"] == "npx_spont/sims/2023_09_19":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )       
        # case npx 10min 32ch hex_O rou:0.3 pfr:0.4    
        # the probe sites are located with respect to hex_01
        # only hex_0 is simulated
        if data_conf["date"] == "npx_spont/sims/2023_10_01":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_32ch_hex_O(
                WEIGHT_PATH, nsites - 1
            )
        # the probe sites are located with respect to hex_01
        # not hex_0
        if data_conf["date"] == "npx_spont/sims/2023_10_13":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )
        # the probe sites are located with respect to hex_01
        # not hex_0
        # these are the in silico neuropixels to compare to 
        # in vivo Marques-Smith
        if data_conf["date"] == "npx_spont/sims/2023_10_18":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )
        if data_conf["date"] == "npx_spont/sims/2024_01_30":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )        
        if data_conf["date"] == "npx_spont/sims/2024_02_01":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )        
        if data_conf["date"] == "npx_spont/sims/2024_02_02":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )        
        # the probe sites are located with respect to hex_01
        # not hex_0
        if data_conf["date"] == "npx_evoked":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_npx_384ch(
                WEIGHT_PATH, nsites - 1
            )
    # case horvath in-silico probe 
    elif data_conf["exp"] == "dense_spont":

        # CAMPAIGN 1 ------------

        # the horvath probe at depth 1 campaign 1
        # not hex_0
        if data_conf["date"] == "sims/campaign_1/probe_1":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_1(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 2 campaign 1
        # not hex_0
        if data_conf["date"] == "sims/campaign_1/probe_2":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_2(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 3 campaign 1
        # not hex_0
        if data_conf["date"] == "sims/campaign_1/probe_3":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_3(
                WEIGHT_PATH, nsites - 1
            )             
        
        # CAMPAIGN 2 ------------

        # the horvath probe at depth 1 campaign 2
        # not hex_0
        if data_conf["date"] == "sims/campaign_2/probe_1":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_1(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 2 campaign 2
        # not hex_0
        if data_conf["date"] == "sims/campaign_2/probe_2":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_2(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 3 campaign 2
        # not hex_0
        if data_conf["date"] == "sims/campaign_2/probe_3":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_3(
                WEIGHT_PATH, nsites - 1
            )     
        
        # CAMPAIGN 3 ------------

        # the horvath probe at depth 1 campaign 3
        # not hex_0
        if data_conf["date"] == "sims/campaign_3/probe_1":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_1(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 2 campaign 3
        # not hex_0
        if data_conf["date"] == "sims/campaign_3/probe_2":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_2(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 3 campaign 3
        # not hex_0
        if data_conf["date"] == "sims/campaign_3/probe_3":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_3(
                WEIGHT_PATH, nsites - 1
            )            
                                       
        # CAMPAIGN 4 ------------

        # the horvath probe at depth 1 campaign 4
        # not hex_0
        if data_conf["date"] == "sims/campaign_4/probe_1":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_1(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 2 campaign 4
        # not hex_0
        if data_conf["date"] == "sims/campaign_4/probe_2":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_2(
                WEIGHT_PATH, nsites - 1
            )
        # the horvath probe at depth 3 campaign 4
        # not hex_0
        if data_conf["date"] == "sims/campaign_4/probe_3":
            WEIGHT_PATH = data_conf["campaign"]["source_weights"]
            return load_locations_from_weights_silico_horvath_probe_3(
                WEIGHT_PATH, nsites - 1
            )
    # case james experiment
    elif data_conf["exp"] == "others/spikewarp":
        WEIGHT_PATH = data_conf["campaign"]["source_weights"]
        return load_locations_from_weights_npx_384ch(
            WEIGHT_PATH, nsites - 1
        )
    else:
        raise NotImplementedError("""This experiment is not implemented, 
                                  implement it in src/nodes/dataeng/lfp_only/campaign_stacking.py""")


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


def load_locations_from_weights_npx_384ch(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        np.array: contacts' 3D coordinates
    """
    # loading weights file with channel locations
    f = h5py.File(weightspath, "r")

    locationstmp = np.array([0, 0, 0])
    for i in range(nsites):
        temp = np.array(
            f["electrodes"]["Neuropixels-384_" + str(i)]["location"]
        )
        locationstmp = np.c_[locationstmp, temp]
    locations = locationstmp.T[1:]
    return locations


def load_locations_from_weights_npx_32ch_hex_O(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file for npx_384ch_hex_O1
    the probe is centered with respect to  hex_01

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        np.array: contacts' 3D coordinates
    """
    # loading weights file with channel locations
    f = h5py.File(weightspath, "r")

    locationstmp = np.array([0, 0, 0])
    contacts = np.arange(127, 127+32, 1)

    for i in contacts:
        temp = np.array(
            f["electrodes"]["Neuropixels-384_" + str(i)]["location"]
        )
        locationstmp = np.c_[locationstmp, temp]
    locations = locationstmp.T[1:]
    return locations


def load_locations_from_weights_npx_384ch_hex_O1(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file for npx_384ch_hex_O1
    the probe is centered with respect to  hex_01

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        np.array: contacts' 3D coordinates
    """
    # loading weights file with channel locations
    f = h5py.File(weightspath, "r")

    locationstmp = np.array([0, 0, 0])
    for i in range(nsites):
        temp = np.array(
            f["electrodes"]["Neuropixels-384_" + str(i)]["location"]
        )
        locationstmp = np.c_[locationstmp, temp]
    locations = locationstmp.T[1:]
    return locations


def load_locations_from_weights_silico_horvath_probe_1(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file for silico horvath probe 1
    the probe is centered with respect to hex_01

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        np.array: contacts' 3D coordinates
    """
    # loading weights file with channel locations
    f = h5py.File(weightspath, "r")

    locationstmp = np.array([0, 0, 0])
    for i in range(nsites):
        temp = np.array(
            f["electrodes"][str(i)]["position"]
        )
        locationstmp = np.c_[locationstmp, temp]
    locations = locationstmp.T[1:]
    return locations


def load_locations_from_weights_silico_horvath_probe_2(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file for silico horvath probe 2
    the probe is centered with respect to hex_01

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        np.array: contacts' 3D coordinates
    """
    return load_locations_from_weights_silico_horvath_probe_1(weightspath, nsites)


def load_locations_from_weights_silico_horvath_probe_3(weightspath: str, nsites: int):
    """Load the channel site locations from the weights file for silico horvath probe 3
    the probe is centered with respect to hex_01

    Args:
        weightspath (str): _description_
        nsites (int): _description_

    Returns:
        np.array: contacts' 3D coordinates
    """
    return load_locations_from_weights_silico_horvath_probe_1(weightspath, nsites)


def find_close_gids(units, probepos: np.array, radius: int):
    """find units within radius of contacts
    The radius used by default is a 50 micron distance
    from the electrodes.

    Args:
        units (_type_): _description_
        probepos (np.array): contact 3D coordinates
        radius (int): _description_

    Returns:
        _type_: _description_
    """
    cpos = units[list("xyz")]
    tiled = np.tile(probepos, (cpos.shape[0], 1))
    dist = np.linalg.norm(cpos - tiled, axis=1)
    return units[dist < radius]


def read_simulation_spike_and_trace(
    sim_i: int, data_conf: dict, param_conf: dict
):
    """read spikes and trace of a simulation with bluepy
    The radius used by default is a 50 micron distance
    from the electrodes.
    
    Args:
        sim_i (int): simulation id e.g., 0
        data_conf (dict): dictionary of dataset paths
        param_conf (dict): dictionary of parameters

    Returns:
        _type_: _description_
    """
    # get parameters
    RADIUS = param_conf["circuit"]["radius"]

    # set read path
    CAMPAIGN_PATH = data_conf["dataeng"]["campaign"]["input"]

    # set write paths
    CHANNEL_LOC_PATH = data_conf["dataeng"]["campaign"]["output"][
        "channel_location_path"
    ]
    NEAR_PROBE_CELLS_FILE_PATH = data_conf["dataeng"]["campaign"]["output"][
        "near_probe_cells_file_path"
    ]

    # read simulation
    blueconfig_path = os.path.join(os.path.join(CAMPAIGN_PATH, str(sim_i)), "BlueConfig")
    sim = bp.Simulation(blueconfig_path)

    # get report (takes a few secs)
    report = sim.report("lfp")

    # read trace data (mV, gid, channel, time)
    sim_traces = report.get(
        t_start=report.meta["start_time"], t_end=report.meta["end_time"]
    )
    n_contacts = int(sim_traces.shape[1])
    sim_traces.columns = np.arange(n_contacts)

    # load contact coordinates
    locations = load_locations_from_weights(data_conf, n_contacts)

    # find units within RADIUS (50 microns) to contacts 
    # (near-probe units)
    cells = sim.circuit.cells.get({"$target": "hex0"})
    all_close_gids = []
    
    for location in locations:
        close_gids_df = find_close_gids(cells, location, RADIUS)
        close_gids = list(close_gids_df.index)
        all_close_gids = all_close_gids + close_gids
    all_close_gids = list(set(all_close_gids))

    # get matching spikes from near-probe units
    sim_spikes = sim.spikes.get(
        gids=all_close_gids,
        t_start=report.meta["start_time"],
        t_end=report.meta["end_time"],
    )

    # write contacts' coordinates
    if not os.path.exists(CHANNEL_LOC_PATH):
        os.makedirs(CHANNEL_LOC_PATH)
    np.save(CHANNEL_LOC_PATH + "_locations", locations, allow_pickle=True)

    # write list of near-probe units
    pd.DataFrame(all_close_gids).to_pickle(NEAR_PROBE_CELLS_FILE_PATH)
    return sim_spikes, sim_traces


def remove_setup_period(sim_spikes, sim_traces):
    """remove the initial period when the circuit is setup 
    (units are connected, etc) which typically last 1 sec
    This is configured in bbp_workflow's config file 
    "GenerateCampaign...cfg"

    Args:
        sim_spikes (_type_): _description_
        sim_traces (_type_): _description_

    Returns:
        _type_: _description_
    """
    # setup period in ms
    # TODO: collect from config directly
    SETUP_PERIOD_MS = 1000

    ## Spikes
    # find timestamps below SETUP_PERIOD_MS
    ttps_to_drop = sim_spikes.index[sim_spikes.index <= SETUP_PERIOD_MS].tolist()

    # drop these timestamps and adjust following ones
    clean_spikes = sim_spikes.drop(ttps_to_drop)
    clean_spikes.index = clean_spikes.index - SETUP_PERIOD_MS

    ## Traces
    # find trace timestamps below SETUP_PERIOD_MS
    ttps_to_drop = sim_traces.index[sim_traces.index <= SETUP_PERIOD_MS].tolist()

    # drop these timestamps and adjust following ones
    clean_traces = sim_traces.drop(ttps_to_drop)
    clean_traces.index = clean_traces.index - SETUP_PERIOD_MS
    return clean_spikes, clean_traces


def run(data_conf: dict, param_conf: dict, blue_config: dict):
    """stack all simulation files into a campaign spike and trace
    file

    Args:
        data_conf (dict): dictionary of dataset paths
        param_conf (dict): dictionary of parameters
        blue_config (dict): dictionary of simulation parameters
    """

    # get write path
    CAMPAIGN_SPIKE_FILE = data_conf["dataeng"]["campaign"]["output"][
        "spike_file_path"
    ]
    CAMPAIGN_TRACE_FILE = data_conf["dataeng"]["campaign"]["output"][
        "trace_file_path"
    ]

    # Trace and spikes are in the same folder
    CAMPAIGN_FILE_PATH = os.path.dirname(CAMPAIGN_SPIKE_FILE)

    # Simulation files
    SIM_IDS = data_conf["dataeng"]["simulations"]["ids"]
    
    # get lfp and spike timesteps between two acquisitions
    # (sampling period)
    LFP_DT = float(blue_config.Report_lfp["Dt"])
    SPIKE_DT = float(blue_config["Run_Default"]["Dt"])

    # get simulation ids
    # simulation_id = get_simulation_ids(CAMPAIGN_PATH)

    # count simulations
    # n_sim = len(simulation_id)

    # initialize
    spike_list = []
    trace_list = []
    spike_current_duration = 0
    lfp_current_duration = 0

    # log
    logger.info("Started stacking simulations for lfp traces and spikes ...")

    # stack simulation spike into campaign level
    for sim_i in SIM_IDS:

        # read spike and trace with bluepy
        # select near-probe units
        sim_spikes, sim_traces = read_simulation_spike_and_trace(
            sim_i, data_conf, param_conf
        )

        # remove the initial circuit setup period 
        # (typically 1 sec, see workflow config files)
        # and reset the lfp and spike timestamps
        sim_spikes, sim_traces = remove_setup_period(sim_spikes, sim_traces)

        # read spikes and traces
        # track simulation duration and
        # increment all spike and trace times
        sim_spikes.index += spike_current_duration
        sim_traces.index += lfp_current_duration

        # append them
        spike_list.append(sim_spikes)
        trace_list.append(sim_traces)

        # track campaign duration
        spike_current_duration = sim_traces.index[-1] + SPIKE_DT
        lfp_current_duration = sim_traces.index[-1] + LFP_DT

        # report
        logger.info(f"Extracted traces and spikes {sim_i}")

    # stack spikes and traces
    campaign_spike = pd.concat(spike_list)
    campaign_trace = pd.concat(trace_list)

    # write
    if not os.path.isdir(CAMPAIGN_FILE_PATH):
        os.makedirs(CAMPAIGN_FILE_PATH)
    campaign_spike.to_pickle(CAMPAIGN_SPIKE_FILE)
    campaign_trace.to_pickle(CAMPAIGN_TRACE_FILE)

    # log
    logger.info(
        "Spike and lfp simulations were written into one campaign in %s",
        CAMPAIGN_FILE_PATH,
    )
    return {
        "spike": campaign_spike,
        "trace": campaign_trace,
    }
