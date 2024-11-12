"""pipeline to concatenate recordings for in silico neuropixels
to compare to vivo Marques-Smith

Usage:
    activate npx... python env.
"""

import os
import yaml 
import logging
import logging.config
import shutil
import time 
import numpy as np 

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

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


def run():
    """concatenate bbp workflow campaigns into a single recording file and spike file
    """
    # get concatenated experiment's config
    data_cfg, _ = get_config("silico_neuropixels", "npx_spont").values()
    SPIKE_PATH = data_cfg["dataeng"]["campaign"]["output"]["spike_file_path"]
    RECORDING_PATH = data_cfg["dataeng"]["campaign"]["output"]["trace_file_path"]
    RUN_PATHS = data_cfg["dataeng"]["concatenated"]

    # get campaign traces and spike paths
    trace_paths = []
    spike_paths = []
    for _, run_i in enumerate(RUN_PATHS):

        # get campaign config
        data_cfg, param_cfg = get_config("silico_neuropixels", run_i).values()

        # stack its simulation into df and pickle
        stack(data_cfg, param_cfg)

        # record paths of pickled files
        trace_paths.append(data_cfg["dataeng"]["campaign"]["output"]["trace_file_path"])
        spike_paths.append(data_cfg["dataeng"]["campaign"]["output"]["spike_file_path"])
    
    # concatenate
    traces, spikes = concat_campaigns.concat_raw_traces_and_spikes(trace_paths, spike_paths)

    # re/create path and write concatenated files
    parent_path = os.path.dirname(RECORDING_PATH)
    shutil.rmtree(parent_path, ignore_errors=True)
    os.makedirs(parent_path)
    spikes.to_pickle(SPIKE_PATH)
    traces.to_pickle(RECORDING_PATH)
    logger.info(f"done saving concatenated campaigns.")