"""Write fitted and wired recording and sorting for the dense probe at depth 3 to NWB files

author: laquitainesteeve@gmail.com

Usage:  

    sbatch cluster/dandi/write_nwb/fitted/dense_spont_p3.sh

Duration: 15 min

Returns:
    NWB: write nwb file
"""
import os
import warnings
from datetime import datetime
import uuid
from datetime import datetime
from dateutil.tz import tzlocal
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
from src.nodes.dandi import write_nwb

# SETUP CONFIG
data_conf, param_conf = get_config("dense_spont", "probe_3").values()
REC_PATH = data_conf["probe_wiring"]["full"]["output"]
GT_PATH = data_conf["ground_truth"]["full"]["output"]
NWB_PATH = data_conf["probe_wiring_nwb"] # write path

# SET PARAMETERS
params = {
    "subject_id": "005-fitted",
    "session_description": "Biophysical simulation of dense probe at depth 3 in the spontaneous regime. Background noise and gain are fitted to Horvath dataset.",
    "identifier": str(uuid.uuid4()),
    "session_start_time": datetime.now(tzlocal()),
    "experimenter": "Laquitaine Steeve",
    "lab": "Blue Brain Project",
    "institution": "EPFL",
    "experiment_description": "Biophysical simulation of dense probe at depth 3 in the spontaneous regime. Background noise and gain are fitted to Horvath dataset.",
    "session_id": "005",
    "related_publications": "doi:"
    }
    
# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# run 
write_nwb.run(REC_PATH, GT_PATH, NWB_PATH, params)