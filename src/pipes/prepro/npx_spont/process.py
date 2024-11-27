"""pipeline to process the "silico_neuropixels" experiment - run "concatenated" 
(simulation of Marques probe) from raw bbp_workflow simulation files to ready-to-sort 
SpikeInterface's Recording and Sorting Extractors

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023

 usage:

    sbatch cluster/prepro/npx_spont/process.sh
    
    or 
    
    source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/envs/spikinterf0_100_5/bin/activate
    python3.9 -c "from src.pipes.prepro.npx_spont.process import run; run(filtering='butterworth')"
    
duration: 3h:54
"""

import os
import logging
import logging.config
import logging.config
import yaml
import time 
import numpy as np
import shutil

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

# custom package
from src.nodes.utils import get_config
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth

# pipeline
from src.pipes.prepro.npx_spont.supp import concat

# SETUP PARAMETERS
data_conf, param_conf = get_config("silico_neuropixels", "npx_spont").values()

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PIPELINE
STACK = True           # done once then set to False
SAVE_REC_EXTRACTOR = False
TUNE_FIT = False         # tune fitted noise
FIT_CAST = False         # done once then set to False (2h18 min)
OFFSET = False
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = False             # done once then set to False (25 mins)
SAVE_METADATA = False    # True to add new metadata to wired probe
PREPROCESS = False      # True to update after adding new metadata (butterworth: 1h40, wavelet: 3h)
GROUND_TRUTH = True    # done once then set to False

# SETUP PARALLEL PROCESSING
# required, else fit and cast as extractor crashes due to lack of 
# memory
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True} # butterworth


def stack():
    """concatenate bbp-workflow simulations into campaigns/experiments
    and experiments into a single recording

    takes 30 min
    """
    concat.run()
    

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
    """
    args:
        filtering: "butterworth" or "wavelet"
    """
    
    # track time
    t0 = time.time()

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
                              blueconfig=data_conf["dataeng"]["blueconfig"],
                              save_metadata=SAVE_METADATA,
                              job_dict=job_dict,
                              n_sites=384,
                              load_atlas_metadata=False,
                              load_filtered_cells_metadata=False)
    
    if PREPROCESS:
        preprocess.preprocess_recording_npx_probe(data_conf=data_conf, 
                                                  param_conf=param_conf, 
                                                  job_dict=job_dict, 
                                                  filtering=filtering)
    
    if GROUND_TRUTH:
        preprocess.save_ground_truth(data_conf, param_conf)

    # report time
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")