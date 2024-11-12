"""pipeline for data processing of horvath silico concatenated recordings with probe 1, 2 or 3
from raw bbp-workflow simulation files to single traces and spikes concatenated over
campaigns

usage:

    sbatch cluster/prepro/dense_spont/process_probe1.sh # (16m)
    sbatch cluster/prepro/dense_spont/process_probe2.sh # (13m)
    sbatch cluster/prepro/dense_spont/process_probe3.sh # (19m)

    or 
    
    source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate
    python3.9 -c "from src.pipes.prepro.dense_spont.process import run; run(experiment='dense_spont', run='probe_1', noise_tuning=1)"
    
duration: takes 1h on a compute node
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

# custom package
from src.nodes.utils import get_config
#from src.nodes.dataeng.silico import campaign_stacking
from src.nodes.prepro import preprocess

# pipelines
from src.pipes.prepro.dense_spont import concat


# SETUP PIPELINE
#STACK_SIM = False
STACK = False
SAVE_REC_EXTRACTOR = False
FIT_CAST = False
OFFSET = False
TUNE_FIT = False        # tune fitted noise
FIT_CAST = False        # done once then set to False (2h18 min)
OFFSET = False
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = False
SAVE_METADATA = False
PREPROCESS = False
GROUND_TRUTH = True


# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PARALLEL PROCESSING
job_dict = {"n_jobs": 1, "progress_bar": True}


# def stack_raw_sims(data_conf, blue_config):
#     """Start from BBP-workflow raw simulations
#     into pickled files

#     Args:
#         data_conf (_type_): _description_
#         blue_config (_type_): _description_
#     """
    
#     campaign_stacking.run(data_conf, blue_config)
    
def stack(experiment: str, run: str):
    """Starts from pickled files and concatenate 
    simulations into campaigns/experiments
    then experiments into a single recording

    takes 30 min
    """
    try:
        concat.run(experiment, run)
    except:
        raise NotImplementedError("This run is not implemented")

    
def tune_fit(data_conf, noise_tuning):
    """manually tune the best fit noise RMS
    for each layer

    Args:
        data_conf (_type_): _description_
    """
    # path
    FITTED_PATH = data_conf["preprocessing"]["fitting"]["fitted_noise"]
    TUNED_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    
    if os.path.isfile(FITTED_PATH + "L1.npy"):
        l1_out = np.load(FITTED_PATH + "L1.npy", allow_pickle=True).item()
        l1_out["missing_noise_rms"] += noise_tuning
        np.save(TUNED_PATH + "L1.npy", l1_out)
    if os.path.isfile(FITTED_PATH + "L2_3.npy"):
        l23_out = np.load(FITTED_PATH + "L2_3.npy", allow_pickle=True).item()
        l23_out["missing_noise_rms"] += noise_tuning
        np.save(TUNED_PATH + "L2_3.npy", l23_out)
    if os.path.isfile(FITTED_PATH + "L4.npy"):
        l4_out = np.load(FITTED_PATH + "L4.npy", allow_pickle=True).item()
        l4_out["missing_noise_rms"] += noise_tuning
        np.save(TUNED_PATH + "L4.npy", l4_out)
    if os.path.isfile(FITTED_PATH + "L5.npy"):
        l5_out = np.load(FITTED_PATH + "L5.npy", allow_pickle=True).item()
        l5_out["missing_noise_rms"] += noise_tuning
        np.save(TUNED_PATH + "L5.npy", l5_out)
    if os.path.isfile(FITTED_PATH + "L6.npy"):
        l6_out = np.load(FITTED_PATH + "L6.npy", allow_pickle=True).item()
        l6_out["missing_noise_rms"] += noise_tuning
        np.save(TUNED_PATH + "L6.npy", l6_out)
    

def run(experiment: str, run: str, noise_tuning):
    """_summary_

    Args:
        run_i (str): e.g., concatenated/probe_1
        experiment (str, optional): _description_. Defaults to "silico_horvath".
    """

    # get config
    data_conf, param_conf = get_config(experiment, run).values()
    logger.info(f"Checking parameters: exp={experiment} and run={run}")

    # track time
    t0 = time.time()
    
    # if STACK_SIM:
    #     stack_raw_sims(data_conf, blue_config)
    
    if STACK:
        stack(experiment, run)
        
    if SAVE_REC_EXTRACTOR:
        preprocess.save_raw_rec_extractor(data_conf, param_conf, job_dict)
        
    if TUNE_FIT:
        tune_fit(data_conf, noise_tuning)
        
    if FIT_CAST:
        Recording = preprocess.fit_and_cast_as_extractor_dense_probe(data_conf=data_conf,
                                                                     offset=OFFSET,
                                                                     scale_and_add_noise=SCALE_AND_ADD_NOISE)
        
    if WIRE:
        preprocess.wire_probe(data_conf=data_conf,
                              param_conf=param_conf,
                              Recording=Recording,
                              blueconfig=None,
                              save_metadata=SAVE_METADATA,
                              job_dict=job_dict,
                              n_sites=128,
                              load_atlas_metadata=False)
        
    if PREPROCESS:
        preprocess.preprocess_recording_dense_probe(data_conf=data_conf,
                                                    param_conf=param_conf,
                                                    job_dict=job_dict)
        
    if GROUND_TRUTH:
        preprocess.save_ground_truth(data_conf, param_conf)
        
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
