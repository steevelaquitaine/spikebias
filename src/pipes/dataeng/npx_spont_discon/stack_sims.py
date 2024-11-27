"""Concatenate the simulation files of each of the three campaigns
run for the npx_spon_disconnected experiment

Usage:

    sbatch cluster/dataeng/npx_spont_discon/sims/process.sh

Takes 10 mins

Returns:

    spiketrain.pkl (pandas series) and trace.pkl (pandas dataframe) files of 
"""

import os
import logging
import logging.config
import logging.config
import yaml
import time 
import numpy as np


# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.lfp_only import stacking



# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def stack(data_conf, param_conf):

    # takes 7 min (for 5 min rec)
    t0 = time.time()

    # get campaign params and stack
    campaign_params = load_campaign_params(data_conf)
    stacking.run(data_conf, param_conf, campaign_params["blue_config"])
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    

def run():
    
    # stack files of campaign1 -
    data_conf, param_conf = get_config("silico_neuropixels", "npx_spont_discon/sims/2024_11_16_disconnected").values()
    stack(data_conf, param_conf)
    
    # stack files of campaign2
    data_conf, param_conf = get_config("silico_neuropixels", "npx_spont_discon/sims/2024_11_16_disconnected_campaign2").values()
    stack(data_conf, param_conf)
    
    # stack files of campaign3
    data_conf, param_conf = get_config("silico_neuropixels", "npx_spont_discon/sims/2024_11_24_disconnected_campaign3").values()
    stack(data_conf, param_conf)