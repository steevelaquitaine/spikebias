"""pipeline to concatenate recordings with probe 1 for in silico Horvath

Usage:
    activate npx... python env.
    python3.9 -m src.pipes.prepro.concat_silico_horvath_probe_1
"""

import os
import pandas as pd
import yaml 
import logging
import logging.config
import shutil
import numpy  as np
import time

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

# SETUP PIPELINE
BUILD_CAMPAIGN = False   # concatenate 50 secs simulations into recording spike and trace files


def stack(data_cfg, param_cfg):
    """Stack bbp_workflow raw simulations into a single pandas dataframe
    and pickle it.
    Returns:
        (pd.DataFrame):
        - value: voltage
        - index: timepoints in ms
        - cols: recording sites
    """

    # takes 7 min (for 5 min rec)
    t0 = time.time()

    # get campaign params and stack
    campaign_params = load_campaign_params(data_cfg)
    stacking.run(data_cfg, param_cfg, campaign_params["blue_config"])
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(experiment: str, run: str):
    """concatenate 10 minutes of blue brain simulation 
    campaigns into a single recording file and spike file
    
    """
    # get config
    data_conf_concat, _ = get_config(experiment, run).values()
    CONCAT_SPIKE_PATH = data_conf_concat["dataeng"]["campaign"]["output"]["spike_file_path"]
    CONCAT_RECORDING_PATH = data_conf_concat["dataeng"]["campaign"]["output"]["trace_file_path"]
    RUN_PATHS = data_conf_concat["dataeng"]["concatenated"]

    # get campaign trace and spike paths
    raw_trace_paths = []
    raw_spike_paths = []
    for ix, run_i in enumerate(RUN_PATHS):
        data_conf_i, _ = get_config(experiment, run_i).values()
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