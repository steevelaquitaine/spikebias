"""pipeline to process the "silico_neuropixels" experiment - run "stimulus" 
(simulation of Marques probe with stimulus) from raw bbp_workflow simulation files to ready-to-sort 
SpikeInterface's Recording and Sorting Extractors

run: 40m noise ftd gain ftd adj 10 perc less

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023

 usage:

    sbatch cluster/prepro/npx_evoked/process.sh
    
duration: 1h:25 on one node (4h before, accelerated with Pytorch)
    - 50 min up to SAVE_REC_EXTRACTOR
"""

import os
import shutil
import logging
import logging.config
import yaml
import time 
import numpy as np
import warnings
import pandas as pd
import spikeinterface.extractors as se 
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.nodes.dataeng.lfp_only import stacking
from src.nodes.dataeng import dataeng
from src.nodes.prepro import preprocess


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
STACK = True           # done once then set to False (0h:30)
SAVE_REC_EXTRACTOR = True
TUNE_FIT = False        # tune fitted noise
FIT_CAST = False        # done once then set to False (2h18 min)
OFFSET = False
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = False              # done once then set to False
SAVE_METADATA = False     # True to add new metadata to wired probe (3h:40)
PREPROCESS = False        # True to update after adding new metadata (1h:50)
GROUND_TRUTH = False      # done once then set to False (13 min)


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


def wire_probe(
        data_conf: dict, 
        param_conf: dict, 
        Recording, 
        blueconfig, 
        save_metadata: bool,
        job_dict: dict, 
        load_atlas_metadata=True, 
        load_filtered_cells_metadata=True
        ):
    """wire a neuropixels 1.0 probe
    
    takes 12 min (versus 48 min w/o multiprocessing)

    note: The wired Recording Extractor is written via 
    multiprocessing on 8 CPU cores, with 1G of memory per job 
    (n_jobs=8 and chunk_memory=1G)

    to check the number of physical cpu cores on your machine:
        cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l
    
    to check the number of logical cpu cores on your machine:
        nproc
    """

    # track time
    t0 = time.time()
    logger.info("Starting ...")

    # get write path
    WRITE_PATH = data_conf["probe_wiring"]["full"]["output"]

    # run and write
    Recording = probe_wiring.run(Recording, data_conf, param_conf, 
                                 load_filtered_cells_metadata=load_filtered_cells_metadata)

    # save metadata
    if save_metadata:
        Recording = set_property_layer(data_conf=data_conf, Recording=Recording,
                                       blueconfig=blueconfig, n_sites=384,
                                       load_atlas_metadata=load_atlas_metadata)

    # write (parallel processing works for 10 min recordings, else use 1 node for 1h recording otherwise
    # you get "out of memory error: "slurmstepd: error: Detected 50 oom-kill event(s). 
    # Some of your processes may have been killed by the cgroup out-of-memory handler."")
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Recording.save(folder=WRITE_PATH, format="binary", **job_dict)        
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def preprocess_recording(job_dict: dict, filtering='butterworth'):
    """preprocess recording

    takes 15 min (vs. 32 min w/o multiprocessing)
    """

    # write path
    WRITE_PATH = data_conf["preprocessing"]["full"]["output"]["trace_file_path"]

    # takes 32 min
    t0 = time.time()
    logger.info("Starting 'preprocess_recording'")
    
    # preprocess
    Preprocessed = preprocess.run_butterworth_filtering_noise_ftd_gain_ftd_adj10perc_less(data_conf,
                                  param_conf)
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def extract_ground_truth():

    # set paths
    WRITE_PATH = data_conf["ground_truth"]["full"]["output"]
    
    # takes about 3 hours
    t0 = time.time()
    logger.info("Starting 'extract_ground_truth'")

    # get simulation parameters
    simulation = load_campaign_params(data_conf)

    # cast ground truth spikes as a SpikeInterface Sorting Extractor object (1.5h for 534 units)
    SortingTrue = ground_truth.run(simulation, data_conf, param_conf)

    # write
    t0 = time.time()
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    SortingTrue["ground_truth_sorting_object"].save(folder=WRITE_PATH, n_jobs=-1, total_memory="2G")
    logger.info("Done writting Ground truth SortingExtractor in %s", round(time.time()-t0, 1))


def run(filtering: str="wavelet"):
    
    # track time
    t0 = time.time()
    
    # run pipeline
    if STACK:
        stack()
        
    if SAVE_REC_EXTRACTOR:
        dataeng.save_raw_rec_extractor(data_conf)
        
    if TUNE_FIT:
        tune_fit(data_conf)
        
    if FIT_CAST:
        Recording = dataeng.fit_and_cast_as_extractor(data_conf, OFFSET, SCALE_AND_ADD_NOISE)
        
    if WIRE:
        preprocess.wire_probe(data_conf=data_conf,
                              param_conf=param_conf,
                              Recording=Recording,
                              blueconfig=data_conf["dataeng"]["blueconfig"], # None
                              save_metadata=SAVE_METADATA,
                              job_dict=job_dict, 
                              n_sites=384,
                              load_atlas_metadata=False, # False
                              load_filtered_cells_metadata=False) # False     
           
    if PREPROCESS:
        preprocess_recording(job_dict, filtering)
        
    if GROUND_TRUTH:
        extract_ground_truth()

    # report time
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")