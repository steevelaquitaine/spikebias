"""
  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023
modified: 11.03.2024

 usage:

    sbatch cluster/processing/marques_silico/process_40m_noise_none_gain_1e5.sbatch

Note:
    - if preprocessing write crashes because of memory issue. Rerun with all pipeline nodes
    set to False except PREPROCESS=True
    - the trace array requires 240 GB RAM (free RAM is typically 636 GB on a compute core)

Duration: 
    3h:03


References:
    https://spikeinterface.readthedocs.io/en/latest/modules/core.html?highlight=%22total_memory%22#parallel-processing-and-job-kwargs
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np
import shutil 
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface as si

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
#from src.nodes.dataeng.lfp_only import stacking
from src.pipes.metadata.marques_silico import label_layers
from src.pipes.prepro.marques_silico.supp import concat

# SETUP PARAMETERS
data_conf, param_conf = get_config("silico_neuropixels", "npx_spont").values()
BLUECONFIG = data_conf["dataeng"]["blueconfig"]

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PIPELINE
STACK = False           # done once then set to False
FIT_CAST = True         # done once then set to False (2h18 min)
OFFSET = True
SCALE_AND_ADD_NOISE = 1e5
WIRE = True             # done once then set to False (1h06)
SAVE_METADATA = True    # True to add new metadata to wired probe
PREPROCESS = True       # True to update after adding new metadata (butterworth: 1h40, wavelet: 3h)
GROUND_TRUTH = False    # done once then set to False

SFREQ = 40000

# PATHS



# SETUP PARALLEL PROCESSING
# required, else fit and cast as extractor crashes due to lack of 
# memory
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True} # butterworth
# job_dict = {"n_jobs": 1, "chunk_memory": "4G", "progress_bar": True} # wavelet (validated)
# job_dict = {"n_jobs": 4, "chunk_memory": "40G", "progress_bar": True} # fails

def _save_metadata(Recording, blueconfig:str):
    """Save layer metadata to the Recording Extractor
    """
    return label_layers(Recording, blueconfig)


def stack():
    """concatenate bbp-workflow simulations into campaigns/experiments
    and experiments into a single recording

    takes 30 min
    """
    concat.run()


def fit_and_cast_as_extractor():
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
    Recording = recording.run(data_conf, offset=OFFSET, scale_and_add_noise=SCALE_AND_ADD_NOISE)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording


def wire_probe(Recording, save_metadata:bool, job_dict=job_dict):
    """wire a neuropixels 1.0 probe and write
    
    takes 12 min (versus 48 min w/o multiprocessing)

    note: The wired Recording Extractor is written via 
    multiprocessing on 8 CPU cores, with 1G of memory per job 
    (n_jobs=8 and chunk_memory=1G)

    to check the number of physical cpu cores on your machine:
        cat /proc/cpuinfo | grep 'physical id' | sort -u | wc -l
    
    to check the number of logical cpu cores on your machine:
        nproc
    """
    logger.info("Starting ...")
    
    # path
    WRITE_PATH = data_conf["probe_wiring"]["40m"]["output_noise_none_gain_1e5_int16"]

    # track time
    t0 = time.time()

    # run and write
    Recording = probe_wiring.run(Recording, data_conf, param_conf)

    # save metadata
    if save_metadata:
        Recording = _save_metadata(Recording, BLUECONFIG)

    # write (parallel processing works for 10 min recordings, else use 1 node for 1h recording otherwise
    # you get "out of memory error: "slurmstepd: error: Detected 50 oom-kill event(s). 
    # Some of your processes may have been killed by the cgroup out-of-memory handler."")
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Recording.save(folder=WRITE_PATH, format="binary", **job_dict)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording

def preprocess_recording(Wired, job_dict: dict, filtering='butterworth', load_wired=False):
    """preprocess recording and write

    Args:   
        Recording: 
        job_dict
        filtering: 'butterworth' or 'wavelet'

    takes 15 min (vs. 32 min w/o multiprocessing)
    """
    logger.info("Starting 'preprocess_recording'")

    # write path
    READ_PATH = data_conf["probe_wiring"]["40m"]["output_noise_none_gain_1e5_int16"]
    WRITE_PATH = data_conf["preprocessing"]["output"]["40m"]["trace_file_path_noise_none_gain_1e5_int16"]

    # track time
    t0 = time.time()

    # case load
    if load_wired:
        Wired = si.load_extractor(READ_PATH)
    
    # preprocess and write
    Preprocessed = preprocess.run_noise_0uV_gain_x(Wired,
                                                   data_conf,
                                                   param_conf,
                                                   filtering=filtering)
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Preprocessing done in {np.round(time.time()-t0,2)} secs")


def extract_ground_truth():

    # takes about 3 hours
    t0 = time.time()
    logger.info("Starting 'extract_ground_truth'")

    # get simulation parameters
    simulation = load_campaign_params(data_conf)

    # cast ground truth spikes as a SpikeInterface Sorting Extractor object (1.5h for 534 units)
    SortingTrue = ground_truth.run(simulation, data_conf, param_conf)

    # write
    ground_truth.write(SortingTrue["ground_truth_sorting_object"], data_conf)   
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(filtering:str="wavelet"):
    """
    args:
        filtering: "butterworth" or "wavelet"
    """
    
    # track time
    t0 = time.time()

    # run pipeline
    if STACK:
        stack()
    if FIT_CAST:
        Recording = fit_and_cast_as_extractor()
    if WIRE:
        Recording = wire_probe(Recording, save_metadata=SAVE_METADATA, job_dict=job_dict)
    if PREPROCESS:
        preprocess_recording(Recording, job_dict, filtering)
    if GROUND_TRUTH:
        extract_ground_truth()

    # report time
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")

#run()