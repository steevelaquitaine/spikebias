"""Download NWB file

author: laquitainesteeve@gmail.com
Description: download NWB dataset for the dense probe at depth 2

Usage:  

    sh cluster/nwb/download/dense_spont_p1.sh
    
    or
    
    # activate dandi2 python environment
    source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/dandi2/bin/activate
    python -m src.pipes.nwb.download.dense_spont_p2

Returns:
    _type_: _description_
"""
# SETUP PACKAGES 
import os
import numpy as np
import warnings
from pynwb import NWBHDF5IO
import logging
import logging.config
import yaml

warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=UserWarning) 

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config

# SETUP CONFIG
data_conf, param_conf = get_config("dense_spont", "probe_2").values()
NWB_PATH = data_conf["nwb"]

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# dataset id parameters
DANDISET_ID  = "001250"
subject_file = "sub-004/"

# download to path
logger.info("Downloading NWB file...")
os.chdir(os.path.dirname(NWB_PATH))
os.system(f"dandi download https://dandiarchive.org/dandiset/{DANDISET_ID}/draft/files?location={subject_file}")
logger.info("DONE writing NWB file.")