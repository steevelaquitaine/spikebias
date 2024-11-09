"""pipeline for data processing of horvath silico concatenated recordings with probe 1, 2 or 3
from raw bbp-workflow simulation files to single traces and spikes concatenated over
campaigns

usage:

    sbatch cluster/prepro/dense_spont/process_probe1.sh # (16m)
    sbatch cluster/prepro/dense_spont/process_probe2.sh # (13m)
    sbatch cluster/prepro/dense_spont/process_probe3.sh # (19m)

duration: takes 1h on a compute node
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import shutil

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.load import load_campaign_params
from src.nodes.dataeng.silico import recording, probe_wiring
from src.nodes.prepro import preprocess
from src.nodes.truth.silico import ground_truth
#from src.nodes.dataeng.silico import campaign_stacking
from src.nodes.dataeng.dataeng import save_raw_rec_extractor

from src.pipes.prepro.dense_spont import concat
from src.pipes.metadata.dense_spont import label_layers


# SETUP PIPELINE
#STACK_SIM = False
STACK = True
SAVE_REC_EXTRACTOR = True
FIT_CAST = False
OFFSET = False
TUNE_FIT = False        # tune fitted noise
FIT_CAST = False        # done once then set to False (2h18 min)
OFFSET = False
SCALE_AND_ADD_NOISE = {"gain_adjust": 0.90}
WIRE = False
SAVE_METADATA = False
PREPROCESS = False
GROUND_TRUTH = False


# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


# SETUP PARALLEL PROCESSING
job_dict = {"n_jobs": 1, "progress_bar": True}


def _save_metadata(Recording, blueconfig: str):
    """Save layer metadata to the Recording Extractor
    """
    return label_layers(Recording, blueconfig)


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
    
    
def fit_and_cast_as_extractor(data_conf):
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

    # remove 129th "test" channel (actually 128 because starts at 0)
    if len(Recording.channel_ids) == 129:
        Recording = Recording.remove_channels([128])
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording   


def wire_probe(
        data_conf, param_conf, Recording, blueconfig, save_metadata: bool,
        job_dict: dict
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

    # run and write
    Recording = probe_wiring.run(Recording, data_conf, param_conf)

    # save metadata
    if save_metadata:
        Recording = _save_metadata(Recording, blueconfig)

    # write
    probe_wiring.write(Recording, data_conf, job_dict)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def preprocess_recording(data_conf, param_conf):
    """preprocess recording

    takes 15 min (vs. 32 min w/o multiprocessing)
    """

    # takes 32 min
    t0 = time.time()
    logger.info("Starting 'preprocess_recording'")

    # preprocess, write
    Preprocessed = preprocess.run(data_conf, param_conf)
    preprocess.write(Preprocessed, data_conf, job_dict)

    # sanity check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")

#/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataeng/0_silico/silico_horvath_probe2_lfp_10m_128ch_hex0_rou04_pfr03_20Khz_2023_10_28/de7adc53-05d5-44f7-ad3e-9e92c575c7fc/campaign/raw/traces.pkl

def extract_ground_truth(data_conf, param_conf):

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


def run(experiment: str, run: str, noise_tuning):
    """_summary_

    Args:
        run_i (str): e.g., concatenated/probe_1
        experiment (str, optional): _description_. Defaults to "silico_horvath".
    """


    # get config
    data_conf, param_conf = get_config(experiment, run).values()
    logger.info(f"Checking parameters: exp={experiment} and run={run}")

    # set paths
    BLUECONFIG = data_conf["dataeng"]["blueconfig"]

    # track time
    t0 = time.time()
    # if STACK_SIM:
    #     stack_raw_sims(data_conf, blue_config)
    if STACK:
        stack(experiment, run)
    if SAVE_REC_EXTRACTOR:
        save_raw_rec_extractor(data_conf)
    if TUNE_FIT:
        tune_fit(data_conf, noise_tuning)  
    if FIT_CAST:
        Recording = fit_and_cast_as_extractor(data_conf)
    if WIRE:
        wire_probe(data_conf, param_conf, Recording, BLUECONFIG,
                   save_metadata=SAVE_METADATA, job_dict=job_dict)
    if PREPROCESS:
        preprocess_recording(data_conf, param_conf)
    if GROUND_TRUTH:
        extract_ground_truth(data_conf, param_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
