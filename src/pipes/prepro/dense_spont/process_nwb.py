"""pipeline for preprocessing the dense probe's simulated recordings

usage:

    sbatch cluster/prepro/dense_spont/process_probe1_nwb.sh
    sbatch cluster/prepro/dense_spont/process_probe2_nwb.sh
    sbatch cluster/prepro/dense_spont/process_probe3_nwb.sh

duration: takes about 1h on a compute node

Regression-testing: 07.11.2024 - OK
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np
import shutil
import spikeinterface.extractors as se

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth

#from src.pipes.metadata.dense_spont import label_layers
from src.nodes.prepro.preprocess import label_layers

# SETUP PIPELINE
FIT_CAST = True
OFFSET = True
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = True
SAVE_METADATA = True
PREPROCESS = True
GROUND_TRUTH = True


# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PARALLEL PROCESSING
job_dict = {"n_jobs": 1, "progress_bar": True}


def _save_metadata(data_conf, Recording, blueconfig=None, load_atlas_metadata=True):
    """Save layer metadata to the Recording Extractor
    """
    return label_layers(data_conf, Recording, blueconfig, n_sites=128,
                        load_atlas_metadata=load_atlas_metadata)


def fit_and_cast_as_extractor(data_conf: dict, param_conf: dict):
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 

    takes 54 min
    note: RecordingExtractor is not dumpable and can't be processed in parallel
    """
    # track time
    t0 = time.time()
    logger.info("Starting ...")

    # cast (30 secs)
    Recording = recording.run_from_nwb(data_conf, param_conf, offset=OFFSET, scale_and_add_noise=SCALE_AND_ADD_NOISE)

    # remove 129th "test" channel (actually 128 because starts at 0)
    if len(Recording.channel_ids) == 129:
        Recording = Recording.remove_channels([128])
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording   


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
    """wire a the dense probe (from Horvath et al.)
    128 electrodes
    
    Args:
        data_conf (dict):
        param_conf (dict):
        Recording (RecordingExtractor):
        blueconfig (None): DEPRECATED should always be None
        save_metadata (bool)
        job_dict: dict, 
        load_atlas_metadata (Boolean): True: loads existing metadata, else requires the Atlas (download on Zenodo)
        load_filtered_cells_metadata: True: loads existing metadata; can only be true
        
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
    Recording = probe_wiring.run(Recording, data_conf, 
                                 param_conf, load_filtered_cells_metadata)

    # save metadata
    if save_metadata:
        Recording = _save_metadata(data_conf, Recording, blueconfig, 
                                   load_atlas_metadata=load_atlas_metadata)

    # write (parallel processing works for 10 min recordings, else use 1 node for 1h recording otherwise
    # you get "out of memory error: "slurmstepd: error: Detected 50 oom-kill event(s). 
    # Some of your processes may have been killed by the cgroup out-of-memory handler."")
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Recording.save(folder=WRITE_PATH, format="binary", **job_dict)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def preprocess_recording(data_conf, param_conf, job_dict):
    """preprocess recording

    takes 15 min (w/ multiprocessing, else 32 mins)
    """

    # track time
    t0 = time.time()
    logger.info("Starting 'preprocess_recording'")
    
    # write path
    WRITE_PATH = data_conf["preprocessing"]["full"]["output"]["trace_file_path"]
    
    # preprocess, write
    Preprocessed = preprocess.run(data_conf, param_conf)
    
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def extract_ground_truth(data_conf):

    # takes about 3 hours
    t0 = time.time()
    logger.info("Starting 'extract_ground_truth'")

    # write path
    NWB_PATH = data_conf["nwb"]
    
    # load ground truth
    SortingTrue = se.NwbSortingExtractor(NWB_PATH)

    # save
    ground_truth.write(SortingTrue, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(experiment: str, run: str):
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
    
    if FIT_CAST:
        Recording = fit_and_cast_as_extractor(data_conf, 
                                              param_conf)
    
    if WIRE:
        wire_probe(data_conf,
                   param_conf,
                   Recording,
                   blueconfig=None, # data_conf["dataeng"]["blueconfig"], # None
                   save_metadata=SAVE_METADATA,
                   job_dict=job_dict, 
                   load_atlas_metadata=True, # False
                   load_filtered_cells_metadata=True) # False
    
    if PREPROCESS:
        preprocess_recording(data_conf, param_conf, job_dict)
    
    if GROUND_TRUTH:
        extract_ground_truth(data_conf)
        
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")