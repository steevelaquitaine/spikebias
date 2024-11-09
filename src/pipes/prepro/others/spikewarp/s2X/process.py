"""pipeline to process 2X-accelerated neuropixels 
evoked biophysical model for James Isbister's Spike 
time warping project.

- full recording
- fitted and tuned noise and gain
- 2X-accelerated

  author: steeve.laquitaine@epfl.ch
    date: 20.09.2024

 usage:
    
    ## 1. (once only) wire original recording (w/o preprocessing)
    # sbatch cluster/prepro/others/spikewarp/process.sh
    
    # 2. speed-up and preprocess
    sbatch cluster/prepro/others/spikewarp/2X/process.sh

Note:
    - if preprocessing write crashes because of memory issue. Rerun with all pipeline nodes
    set to False except PREPROCESS=True
    - the trace array requires 240 GB RAM (we typically have 636 GB available RAM on our compute node)

Duration: 24 minutes for 1 hour of recording

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
import spikeinterface as si
from spikeinterface import NumpyRecording

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.pipes.metadata.marques_silico import label_layers
from src.pipes.prepro.marques_silico.supp import concat

# SETUP PARAMETERS
cfg, param_conf = get_config("others/spikewarp", "2X").values()
BLUECONFIG = cfg["dataeng"]["blueconfig"]

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PIPELINE
SPEED_2X = True         # 2X acceleration by taking every second sample
PREPROCESS = True       # True to update after adding new metadata (butterworth: 1h40, wavelet: 3h)
GROUND_TRUTH_2X = True  # done once then set to False

# SETUP PARALLEL PROCESSING
# required, else fit and cast as extractor crashes due to lack of 
# memory
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True} # butterworth


def sample2X(Wired, save: bool, job_dict: dict):

    # track time
    t0 = time.time()
    logger.info("Started recording speed-up by 2X ...")
    
    # WRITE PATH
    WIRED_PATH = cfg["probe_wiring"]["full"]["output"]
    
    # downsample traces
    traces = Wired.get_traces()
    new_traces = traces[::2, :]

    # copy parameters
    sf = Wired.get_sampling_frequency()
    cid = Wired.get_channel_ids()

    # create new RecordingExtractor
    Wired_new = NumpyRecording([new_traces], sampling_frequency=sf, channel_ids=cid)

    # copy annotations
    for key in Wired.get_annotation_keys():
        if not key == "is_filtered":
            Wired_new.set_annotation(key, Wired.get_annotation(key))

    # copy properties
    for key in Wired.get_property_keys():
        Wired_new.set_property(key, Wired.get_property(key))

    # copy probe
    if Wired.has_probe():
        probe = Wired.get_probe()
        Wired_new.set_probe(probe)
    
    # save
    if save:
        logger.info("Writing 2X-Wired recording ...")
        shutil.rmtree(WIRED_PATH, ignore_errors=True)
        Wired.save(folder=WIRED_PATH, format="binary", **job_dict)
    
    logger.info(f"Done writting in {np.round(time.time()-t0,2)} secs")
    return Wired_new


def preprocess_recording(job_dict: dict, filtering='butterworth'):
    """preprocess recording and write

    Args:   
        job_dict
        filtering: 'butterworth' or 'wavelet'

    takes 15 min (vs. 32 min w/o multiprocessing)
    """
    # write path
    WRITE_PATH = cfg["preprocessing"]["full"]["output"]["trace_file_path"]

    # takes 32 min
    t0 = time.time()
    logger.info("Starting 'preprocess_recording'")
    
    # preprocess
    Preprocessed = preprocess.run_butterworth_filtering_noise_ftd_gain_ftd_adj10perc_less(cfg,
                                  param_conf)
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def extract_ground_truth2X():

    # takes about 3 hours
    t0 = time.time()
    logger.info("Starting 'extract_ground_truth'")

    # get simulation parameters
    simulation = load_campaign_params(cfg)

    # cast ground truth spikes as a SpikeInterface Sorting Extractor object (1.5h for 534 units)
    SortingTrue = ground_truth.run2X(simulation, cfg, param_conf)

    # save 
    ground_truth.write(SortingTrue["ground_truth_sorting_object"], cfg)   
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(filtering: str = "wavelet"):
    """
    args:
        filtering: "butterworth" or "wavelet"
    """
    
    # track time
    t0 = time.time()
    
    # PATHS
    # original speed
    WIRED_READ = cfg["probe_wiring"]["full"]["input"]
    logger.info("Starting preprocessing...")
    
    # load original wired recording
    Wired = si.load_extractor(WIRED_READ)
    logger.info("Loaded wired recording.")
    
    # run pipeline
    if SPEED_2X:
        Wired = sample2X(Wired, save=True, job_dict=job_dict)
        logger.info("Speeded up wired recording by 2X.")
    if PREPROCESS:
        preprocess_recording(job_dict, filtering)
        logger.info("Preprocessed wired recording.")
    if GROUND_TRUTH_2X:
        extract_ground_truth2X()
        logger.info("Created ground truth SortingExtractor.")

    # report time
    logger.info(f"Preprocessing done for 2X-accelerated recording in {np.round(time.time()-t0,2)} secs")