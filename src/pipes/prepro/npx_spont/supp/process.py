"""pipeline to process the "silico_neuropixels" experiment - run "2023_10_18" 
(simulation of Marques probe) from raw bbp_workflow simulation files to ready-to-sort 
SpikeInterface's Recording and Sorting Extractors

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023
modified: 23.01.2024

 usage:

    sbatch cluster/processing/marques_silico/process.sbatch

duration: 1h:36 (reduced from 2h:20 min with multiprocessing) on a compute node, 
when spike and trace acquisition have same sampling frequencies


References:
    https://spikeinterface.readthedocs.io/en/latest/modules/core.html?highlight=%22total_memory%22#parallel-processing-and-job-kwargs
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
from src.nodes.dataeng.lfp_only import stacking
from src.pipes.metadata.marques_silico import label_layers

# SETUP PARAMETERS
data_conf, param_conf = get_config("silico_neuropixels", "2023_10_18").values()
BLUECONFIG = data_conf["dataeng"]["blueconfig"]

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PIPELINE
STACK = True            # done once then set to False
FIT_CAST = False         # done once then set to False
WIRE = False             # done once then set to False
SAVE_METADATA = False    # True to add new metadata to wired probe
PREPROCESS = False       # True to update after adding new metadata
GROUND_TRUTH = False     # done once then set to False

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
    """

    # takes 7 min (for 5 min rec)
    t0 = time.time()

    # get campaign params and stack
    campaign_params = load_campaign_params(data_conf)
    stacking.run(data_conf, param_conf, campaign_params["blue_config"])
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


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
    Recording = recording.run(data_conf, offset=True, scale_and_add_noise=True)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording


def wire_probe(Recording, save_metadata:bool):
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

    # run and write
    Recording = probe_wiring.run(Recording, data_conf, param_conf)

    # save metadata
    if save_metadata:
        Recording = _save_metadata(Recording, BLUECONFIG)

    # write
    probe_wiring.write(Recording, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")



def preprocess_recording():
    """preprocess recording

    takes 15 min (vs. 32 min w/o multiprocessing)
    """

    # takes 32 min
    t0 = time.time()
    logger.info("Starting 'preprocess_recording'")

    # preprocess, write
    Preprocessed = preprocess.run(data_conf, param_conf)
    preprocess.write(Preprocessed, data_conf)

    # sanity check is preprocessed
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


def run():
    
    # track time
    t0 = time.time()

    # run pipeline
    if STACK:
        stack()
    if FIT_CAST:
        Recording = fit_and_cast_as_extractor()
    if WIRE:
        wire_probe(Recording, save_metadata=SAVE_METADATA)
    if PREPROCESS:
        preprocess_recording()
    if GROUND_TRUTH:
        extract_ground_truth()

    # report time
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")

# run pipeline
run()