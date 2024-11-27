"""Upload all fitted and wired biophysically-simulated datasets to Dandi Archive

author: steeve.laquitaine@epfl.ch; laquitainesteeve@gmail.com

Usage:
    
    sbatch cluster/dandi/upload/fitted/upload.sh
    
    1. Login to dandiarchive
    2. New dandiset - name and description
    3. Copy the **API key** (click on your initial upper right):
    4. Copy the Dataset: click on the link dots to the left of the title, copy its **link**:
        * It should look like: https://gui-staging.dandiarchive.org/#/dandiset/100792

duration:
    - 11 min for the three dense probe recordings
    - just under 8 hours for npx_spont 100 GB raw
    - just 1 hour on the week end for fitted npx_spont and npx_evoked 200 GB total
"""
# nwb software package
# other utils package
import os
import yaml 
import logging
import logging.config

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)
from src.nodes.utils import get_config

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# GET NWB DATASETS PATHS

# NEUROPIXELS BIOPHY  -------------------------------

# spontaneous
cfg_ns, _ = get_config("silico_neuropixels", "npx_spont").values()
NS = os.path.dirname(cfg_ns["probe_wiring_nwb"])

# evoked
cfg_e, _ = get_config("silico_neuropixels", "npx_evoked").values()
NE = os.path.dirname(cfg_e["probe_wiring_nwb"])

# evoked 40 Khz
cfg_e40, _ = get_config("others/spikewarp", "2024_04_13").values()
NE40 = os.path.dirname(cfg_e40["probe_wiring_nwb"])

# evoked 40 Khz
cfg_discon, _ = get_config("silico_neuropixels", "npx_spont_discon").values()
NS_disc = os.path.dirname(cfg_discon["probe_wiring_nwb"])

# NEUROPIXELS VIVO  -------------------------------

# marques-smith data
cfg_ms, _ = get_config("vivo_marques", "c26").values()
MS = os.path.dirname(cfg_ms["probe_wiring_nwb"])


# DENSE SPONT BIOPHY -------------------------------

# dense spontaneous depth 1
cfg_ds1, _ = get_config("dense_spont", "probe_1").values()
DENSE_SP_P1 = os.path.dirname(cfg_ds1["probe_wiring_nwb"])

# dense spontaneous depth 2
cfg_ds2, _ = get_config("dense_spont", "probe_2").values()
DENSE_SP_P2 = os.path.dirname(cfg_ds2["probe_wiring_nwb"])

# dense spontaneous depth 3
cfg_ds3, _ = get_config("dense_spont", "probe_3").values()
DENSE_SP_P3 = os.path.dirname(cfg_ds3["probe_wiring_nwb"])


# DENSE VIVO  -------------------------------

# dense spontaneous depth 1
cfg_hv1, _ = get_config("vivo_horvath", "probe_1").values()
HV1 = os.path.dirname(cfg_hv1["probe_wiring_nwb"])

# dense spontaneous depth 2
cfg_hv2, _ = get_config("vivo_horvath", "probe_2").values()
HV2 = os.path.dirname(cfg_hv2["probe_wiring_nwb"])

# dense spontaneous depth 3
cfg_hv3, _ = get_config("vivo_horvath", "probe_3").values()
HV3 = os.path.dirname(cfg_hv3["probe_wiring_nwb"])


# REYES SPONT BIOPHY -------------------------------

cfg_reyes, _ = get_config("silico_reyes", "2023_01_11").values()
REYES_SUMMED = os.path.dirname(cfg_reyes["probe_wiring_nwb"])
REYES_ISOLATED = os.path.dirname(cfg_reyes["probe_wiring_isolated_cell_nwb"])




# path of the dandiset to upload
DANDISET = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/dandiset"

os.chdir(DANDISET)

# 1. download the dandiset metadata
# 2. move to downloaded folder
# 3. organize files according to dandiset rules
# 4. uploads
logger.info("Uploading NWB files with fitted and wired recordings/sorting to DANDI archive...")

os.system(
    f"""
    export DANDI_API_KEY='210e68743286d64e84743bd8980d5771ef82bf4d';
    dandi download 'https://dandiarchive.org/dandiset/001250/draft';
    cd 001250;
    
    dandi organize {NS} -f dry;
    dandi organize {NS};
    dandi upload
    
    dandi organize {NE} -f dry;
    dandi organize {NE};
    dandi upload
        
    dandi organize {DENSE_SP_P1} -f dry;
    dandi organize {DENSE_SP_P1};
    dandi upload
    
    dandi organize {DENSE_SP_P2} -f dry;
    dandi organize {DENSE_SP_P2};
    dandi upload
    
    dandi organize {DENSE_SP_P3} -f dry;
    dandi organize {DENSE_SP_P3};
    dandi upload

    dandi organize {REYES_SUMMED} -f dry;
    dandi organize {REYES_SUMMED};
    dandi upload    
    
    dandi organize {REYES_ISOLATED} -f dry;
    dandi organize {REYES_ISOLATED};
    dandi upload    
    
    dandi organize {MS} -f dry;
    dandi organize {MS};
    dandi upload        

    dandi organize {NE40} -f dry;
    dandi organize {NE40};
    dandi upload  

    dandi organize {HV1} -f dry;
    dandi organize {HV1};
    dandi upload            
    
    dandi organize {HV2} -f dry;
    dandi organize {HV2};
    dandi upload
    
    dandi organize {HV3} -f dry;
    dandi organize {HV3};
    dandi upload      
    
    dandi organize {NS_disc} -f dry;
    dandi organize {NS_disc};
    dandi upload      
    """
)

logger.info("Done uploading to DANDI archive.")