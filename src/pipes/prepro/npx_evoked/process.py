"""pipeline to process the "silico_neuropixels" experiment - run "stimulus"
(simulation of Marques probe with stimulus) from raw bbp_workflow simulation files to ready-to-sort
SpikeInterface's Recording and Sorting Extractors

run: 40m noise ftd gain ftd adj 10 perc less

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023

 usage:

    sbatch cluster/prepro/npx_evoked/process.sh
    
    or
    
    source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate
    python3.9 -c "from src.pipes.prepro.npx_evoked.process import run; run(filtering='butterworth')"
    
duration:
    - 1h25 on a single node
    - 50 min up to SAVE_REC_EXTRACTOR
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.prepro import preprocess
from src.nodes.dataeng.lfp_only import stacking


# SETUP PARAMETERS
data_conf, param_conf = get_config("silico_neuropixels", "npx_evoked").values()

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PARALLEL PROCESSING
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True}

# SETUP PIPELINE
STACK = False           # done once then set to False (0h:30)
SAVE_REC_EXTRACTOR = False
TUNE_FIT = False        # tune fitted noise
FIT_CAST = False        # done once then set to False (2h18 min)
OFFSET = False
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = False              # done once then set to False
SAVE_METADATA = False     # True to add new metadata to wired probe (3h:40)
PREPROCESS = False        # True to update after adding new metadata (1h:50)
GROUND_TRUTH = True      # done once then set to False (13 min)


def stack():
    """Stack bbp_workflow simulations into a single pandas dataframe
    This is done once.
    
    Returns:
        (pd.DataFrame):
        - value: voltage
        - index: timepoints in ms
        - cols: recording sites
    
    takes 7 min (for 5 min rec)
    """

    # track time
    t0 = time.time()

    # get campaign params and stack
    campaign_params = load_campaign_params(data_conf)
    stacking.run(data_conf, param_conf, campaign_params["blue_config"])
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")

    
def tune_fit(data_conf):
    """manually tune the best fit noise RMS
    for each layer

    Args:
        data_conf (_type_): _description_
    """
    # path
    FITTED_PATH = data_conf["preprocessing"]["fitting"]["fitted_noise"]
    TUNED_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    
    # load fitted noises
    l1_out = np.load(FITTED_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FITTED_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FITTED_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FITTED_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FITTED_PATH + "L6.npy", allow_pickle=True).item()

    # add a few microVolts
    l1_out["missing_noise_rms"] += 0.3
    l23_out["missing_noise_rms"] += 0.5
    l4_out["missing_noise_rms"] += 0.5
    l5_out["missing_noise_rms"] += 0.5
    l6_out["missing_noise_rms"] += 0.5

    # save tuned noise
    np.save(TUNED_PATH + "L1.npy", l1_out)
    np.save(TUNED_PATH + "L2_3.npy", l23_out)
    np.save(TUNED_PATH + "L4.npy", l4_out)
    np.save(TUNED_PATH + "L5.npy", l5_out)
    np.save(TUNED_PATH + "L6.npy", l6_out)


def run(filtering: str="wavelet"):
    
    # track time
    t0 = time.time()
    
    # run pipeline
    if STACK:
        stack()
        
    if SAVE_REC_EXTRACTOR:
        preprocess.save_raw_rec_extractor(data_conf, param_conf, job_dict)
        
    if TUNE_FIT:
        tune_fit(data_conf)
        
    if FIT_CAST:
        Recording = preprocess.fit_and_cast_as_extractor(data_conf=data_conf,
                                                         offset=OFFSET,
                                                         scale_and_add_noise=SCALE_AND_ADD_NOISE)
    if WIRE:
        preprocess.wire_probe(data_conf=data_conf,
                              param_conf=param_conf,
                              Recording=Recording,
                              blueconfig=None,
                              save_metadata=SAVE_METADATA,
                              job_dict=job_dict,
                              n_sites=384,
                              load_atlas_metadata=True,
                              load_filtered_cells_metadata=True)
           
    if PREPROCESS:
        preprocess.preprocess_recording_npx_probe(data_conf=data_conf,
                                                  param_conf=param_conf,
                                                  job_dict=job_dict,
                                                  filtering=filtering)
        
    if GROUND_TRUTH:
        preprocess.save_ground_truth(data_conf, param_conf)

    # report time
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")