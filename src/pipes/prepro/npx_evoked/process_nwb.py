"""pipeline to process the "silico_neuropixels" experiment - run "concatenated" 
(simulation of Marques probe) from raw bbp_workflow simulation files to ready-to-sort 
SpikeInterface's Recording and Sorting Extractors

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023
modified: 05.07.2024

 usage:

    sbatch cluster/prepro/npx_evoked/process.sh

Note:
    - if preprocessing write crashes because of memory issue. Rerun with all pipeline nodes
    set to False except PREPROCESS=True
    - the trace array requires 240 GB RAM (we typically have 636 GB available RAM on our compute node)

Duration: 
    3h:54

References:
    https://spikeinterface.readthedocs.io/en/latest/modules/core.html?highlight=%22total_memory%22#parallel-processing-and-job-kwargs
"""

import os
import logging
import logging.config
import logging.config
import yaml
import time 
import numpy as np
import shutil 
import spikeinterface.extractors as se 

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.pipes.metadata.marques_silico import label_layers

# SETUP PARAMETERS
data_conf, param_conf = get_config("silico_neuropixels", "stimulus").values()
BLUECONFIG = data_conf["dataeng"]["blueconfig"]

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PIPELINE
STACK = False          # done once then set to False
TUNE_FIT = True        # tune fitted noise
FIT_CAST = True        # done once then set to False (2h18 min)
OFFSET = True
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = True             # done once then set to False (25 mins)
SAVE_METADATA = True    # True to add new metadata to wired probe
PREPROCESS = False      # True to update after adding new metadata (butterworth: 1h40, wavelet: 3h)
GROUND_TRUTH = False    # done once then set to False

# SETUP PARALLEL PROCESSING
# required, else fit and cast as extractor crashes due to lack of 
# memory
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True} # butterworth


def _save_metadata(Recording, blueconfig:str):
    """Save layer metadata to the Recording Extractor
    """
    return label_layers(Recording, blueconfig)


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
    Recording = recording.run_from_nwb(data_conf, offset=OFFSET, scale_and_add_noise=SCALE_AND_ADD_NOISE)
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

    # track time
    t0 = time.time()

    # get write path
    WRITE_PATH = data_conf["probe_wiring"]["full"]["output"]

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


def preprocess_recording(job_dict: dict):
    """preprocess recording and write

    Args:
        job_dict
        filtering: 'butterworth' or 'wavelet'

    takes 15 min (vs. 32 min w/o multiprocessing)
    """
    # write path
    WRITE_PATH = data_conf["preprocessing"]["output"]["full"]["trace_file_path"]

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

    # get ground truth sorting extractor
    t0 = time.time()
    logger.info("Starting 'extract_ground_truth'")

    # write
    NWB_PATH = data_conf["nwb"]
    SortingTrue = se.NwbSortingExtractor(NWB_PATH)
    
    # save
    ground_truth.write(SortingTrue, data_conf)   
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(filtering: str="wavelet"):
    """
    args:
        filtering: "butterworth" or "wavelet"
    """
    
    # track time
    t0 = time.time()

    # run pipeline
    if TUNE_FIT:
        tune_fit(data_conf)
    if FIT_CAST:
        Recording = fit_and_cast_as_extractor()
    if WIRE:
        wire_probe(Recording, save_metadata=SAVE_METADATA, job_dict=job_dict)
    if PREPROCESS:
        preprocess_recording(job_dict, filtering)
    if GROUND_TRUTH:
        extract_ground_truth()

    # report time
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")