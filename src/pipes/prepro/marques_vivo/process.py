"""pipeline for data processing of marques c26 in vivo recordings

author: steeve.laquitaine@epfl.ch

usage: 

    sbatch cluster/processing/marques_vivo/process.sbatch

duration: takes 20 mins with parallel processing (instead of 54 min w/o)
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
from src.nodes.dataeng.silico import probe_wiring
from src.nodes.prepro import preprocess
import spikeinterface as si


# SETUP CONFIG
data_conf, param_conf = get_config("vivo_marques", "c26").values()
RECORDING_PATH = data_conf["raw"]
SAMP_FREQ = param_conf["run"]["sampling_frequency"]
N_CONTACTS = param_conf["probe"]["n_contacts"]
DTYPE = param_conf["dtype"]

# SETUP PARALLEL COMPUTING
job_dict = {"n_jobs": -1, "chunk_memory": "4G", "progress_bar": True}

# SETUP PIPELINE
CAST = True
WIRE = True
SAVE_METADATA = True
PREPROCESS = True

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def _save_metadata(Recording):
    """save metadata describing recording sites:
    - site layers
    """
    from src.pipes.metadata.marques_vivo import save_layer_metadata
    return save_layer_metadata(Recording)


def cast_as_RecordingExtractor():
    """Cast as a SpikeInterface RecordingExtractor 
    Rescale, offset, cast as Spikeinterface Recording Extractor object
    Traces need rescaling as the simulation produces floats with nearly all values below an amplitude of 1. 
    As traces are binarized to int16 to be used by Kilosort, nearly all spikes disappear (set to 0).
    return_scale=True does not seem to work as default so we have to rewrite the traces with the new 
    """
    # takes 28 mins
    t0 = time.time()
    Recording = si.read_binary(RECORDING_PATH, sampling_frequency=SAMP_FREQ, num_chan=N_CONTACTS, dtype=DTYPE)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    return Recording


def wire_probe(Recording, save_metadata=bool):
    """wire probe to recording (takes 9 min)
    """
    # run and write
    t0 = time.time()
    Recording = probe_wiring.run(Recording, data_conf, param_conf)

    # save metadata
    # - site layers
    if save_metadata:
        Recording = _save_metadata(Recording)

    # write probe
    probe_wiring.write(Recording, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def preprocess_recording(job_dict):
    """preprocess recording (takes 11 mins)
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


def run():
    """run configured pipeline
    """

    t0 = time.time()
    if CAST:
        Recording = cast_as_RecordingExtractor()
    if WIRE:
        wire_probe(Recording, save_metadata=SAVE_METADATA)
    if PREPROCESS:
        preprocess_recording(job_dict)
    print(f"done in {np.round(time.time()-t0,2)} secs")
    
# run pipeline
run()