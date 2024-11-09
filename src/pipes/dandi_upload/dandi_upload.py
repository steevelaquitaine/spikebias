"""Upload to Dandi archive

author: laquitainesteeve@gmail.com

Usage:

    1. Login to dandiarchive
    2. New dandiset - name and description
    3. Copy the **API key** (click on your initial upper right):
    4. Copy the Dataset: click on the link dots to the left of the title, copy its **link**:
        * It should look like: https://gui-staging.dandiarchive.org/#/dandiset/100792
"""

# nwb software package
# other utils package
from datetime import datetime
import os

# SET PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023"
os.chdir(PROJ_PATH)
from src.nodes.utils import get_config

# SETUP CONFIG

# DENSE SPONT BIOPHY -------------------------------

# dense spontaneous depth 1
cfg_ds1, _ = get_config("dense_spont", "probe_1").values()
DENSE_SP_P1 = os.path.dirname(cfg_ds1["nwb"])

# dense spontaneous depth 2
cfg_ds2, _ = get_config("dense_spont", "probe_2").values()
DENSE_SP_P2 = os.path.dirname(cfg_ds2["nwb"])

# dense spontaneous depth 3
cfg_ds3, _ = get_config("dense_spont", "probe_3").values()
DENSE_SP_P3 = os.path.dirname(cfg_ds3["nwb"])

# path of the dandiset to upload
DANDISET = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dandiset"

os.chdir(DANDISET)

# 1. download the dandiset metadata
# 2. move to downloaded folder
# 3. organize files according to dandiset rules
# 4. upload
os.system(
    f"""
    export DANDI_API_KEY='210e68743286d64e84743bd8980d5771ef82bf4d';
    dandi download 'https://dandiarchive.org/dandiset/001250/draft';
    cd 001250;
    
    dandi organize {DENSE_SP_P1} -f dry;
    dandi organize {DENSE_SP_P1};
    dandi upload
    
    dandi organize {DENSE_SP_P2} -f dry;
    dandi organize {DENSE_SP_P2};
    dandi upload
    
    dandi organize {DENSE_SP_P3} -f dry;
    dandi organize {DENSE_SP_P3};
    dandi upload
    """
)