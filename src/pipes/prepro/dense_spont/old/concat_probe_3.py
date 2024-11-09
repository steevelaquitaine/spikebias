"""pipeline to concatenate recordings with probe 3 for in silico Horvath

Usage:
    activate npx... python env.
    python3.9 -m src.pipes.prepro.concat_silico_horvath_probe_3
"""

import os
import pandas as pd
import yaml 
import logging
import logging.config
import shutil

proj_dir = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023"
os.chdir(proj_dir)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico import concat_campaigns
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.lfp_only import stacking

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PARAMETERS
EXPERIMENT = "silico_horvath"
RUN = "concatenated/probe_3"
data_conf_concat, _ = get_config(EXPERIMENT, RUN).values()
CONCAT_SPIKE_PATH = data_conf_concat["dataeng"]["campaign"]["output"]["spike_file_path"]
CONCAT_RECORDING_PATH = data_conf_concat["dataeng"]["campaign"]["output"]["trace_file_path"]
RUN_PATHS = data_conf_concat["dataeng"]["concatenated"]

# SETUP PIPELINE
BUILD_CAMPAIGN = False   # concatenate 50 secs simulations into recording spike and trace files


def build_campaigns():
    """concatenate (50 secs) simulations into recording spike and trace files
    """
    # loop over bbp-workflow campaigns
    # and concatenate 50 secs simulations into a single 
    # recording trace file and spike file
    for ix, run_i in enumerate(RUN_PATHS):
        logger.info(f"processing {run_i}...")        
        data_conf_i, param_conf_i = get_config(EXPERIMENT, run_i).values()
        campaign_params = load_campaign_params(data_conf_i)
        stacking.run(data_conf_i, param_conf_i, campaign_params["blue_config"])
    logger.info(f"done building all campaigns.")
    

def run():
    """concatenate 10 min campaigns into a single recording file and spike file
    """

    # get campaign trace and spike paths
    raw_trace_paths = []
    raw_spike_paths = []
    for ix, run_i in enumerate(RUN_PATHS):
        data_conf_i, _ = get_config(EXPERIMENT, run_i).values()
        raw_trace_paths.append(data_conf_i["dataeng"]["campaign"]["output"]["trace_file_path"])
        raw_spike_paths.append(data_conf_i["dataeng"]["campaign"]["output"]["spike_file_path"])
    
    # concatenate
    traces, spikes = concat_campaigns.concat_raw_traces_and_spikes(raw_trace_paths, raw_spike_paths)

    # re/create path and write concatenated files
    parent_path = os.path.dirname(CONCAT_RECORDING_PATH)
    shutil.rmtree(parent_path, ignore_errors=True)
    os.makedirs(parent_path)
    spikes.to_pickle(CONCAT_SPIKE_PATH)
    traces.to_pickle(CONCAT_RECORDING_PATH)
    logger.info(f"done saving concatenated campaigns.")


# if BUILD_CAMPAIGN:
#     build_campaigns()
# run()