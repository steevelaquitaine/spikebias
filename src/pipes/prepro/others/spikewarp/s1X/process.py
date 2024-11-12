"""pipeline to process James stimulus simulation (40 Hz)

  author: steeve.laquitaine@epfl.ch
    date: 17.04.2024
modified: 21.09.2024

 usage:

    sbatch cluster/prepro/others/spikewarp/1X/process.sh

  return:
    

duration: 6h:13 on one node

References:
    https://spikeinterface.readthedocs.io/en/latest/modules/core.html?highlight=%22total_memory%22#parallel-processing-and-job-kwargs
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np
import warnings
import shutil
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.nodes.dataeng.lfp_only import stacking
from src.pipes.metadata.marques_silico import label_layers

# SETUP PARAMETERS
data_conf, param_conf = get_config("others/spikewarp", "2024_04_13").values()
BLUECONFIG = data_conf["dataeng"]["blueconfig"]


# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PARALLEL PROCESSING
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True}

# SETUP PIPELINE
STACK = False          # done once then set to False (0h:30)
TUNE_FIT = True        # tune fitted noise
FIT_CAST = True        # done once then set to False (2h18 min)
OFFSET = True
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = True              # done once then set to False
SAVE_METADATA = True     # True to add new metadata to wired probe (3h:40)
PREPROCESS = True        # True to update after adding new metadata (1h:50)
GROUND_TRUTH = True      # done once then set to False (13 min)


def _save_metadata(Recording, blueconfig:str):
    """Save layer metadata to the Recording Extractor
    """
    return label_layers(Recording, blueconfig)


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

    # write (2 mins)
    # recording.write(Recording, data_conf)
    # Recording = recording.load(data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording


def wire_probe(Recording, save_metadata:bool, job_dict:dict):
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
    
    # track time
    t0 = time.time()
    
    # run pipeline
    if STACK:
        stack()
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
