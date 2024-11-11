"""preprocess in vivo dense recordings (Horvath)

Usage: 

    sbatch cluster/prepro/horvath_vivo/process_probe1.sh
    sbatch cluster/prepro/horvath_vivo/process_probe2.sh
    sbatch cluster/prepro/horvath_vivo/process_probe3.sh

duration: 26 min on one node
"""

import os
import time
import numpy as np
import spikeinterface.extractors as se
import logging
import logging.config
import yaml
import shutil 
import spikeinterface as si

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.dataeng.silico import recording
from src.nodes.prepro import preprocess
from src.pipes.metadata.horvath_vivo import label_layers


# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PIPELINE
CAST = False
SAVE_METADATA = True
PREPROCESS = True

# distributed-computing speed up
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True}


def cast_as_RecordingExtractor(data_conf):

    # takes 60 mins (6 min recording)
    t0 = time.time()

    # cast (30 secs)
    RecordingExtr = se.NwbRecordingExtractor(data_conf["raw"])

    # write (2 mins)
    recording.write(RecordingExtr, data_conf)
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def save_metadata(experiment: str, run: str, cfg: dict):
    
    # set read paths
    WIRED_READ = cfg["probe_wiring"]["full"]["input"]
    WIRED_WRITE = cfg["probe_wiring"]["full"]["output"]
    
    # track time
    t0 = time.time()
    Wired = si.load_extractor(WIRED_READ)
    
    # save metadata
    Wired = label_layers(Wired, experiment, run)
    
    # save
    shutil.rmtree(WIRED_WRITE, ignore_errors=True)
    Wired.save(folder=WIRED_WRITE, format="binary")
    logger.info(f"Saving metadata - done in {np.round(time.time()-t0,2)} secs")
    return Wired


def load_wired(cfg):
    
    WIRED_PATH = cfg["probe_wiring"]["full"]["input"]
    return si.load_extractor(WIRED_PATH)
    
    
def preprocess_recording(Wired, cfg, param_conf, job_dict):
    
    # write 
    WRITE_PATH = cfg["preprocessing"]["full"]["output"]["trace_file_path"]
    
    # track time
    t0 = time.time()
    
    # preprocess
    Preprocessed = preprocess.run_butterworth(Wired, cfg, param_conf)
    
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Preprocessing done in {np.round(time.time()-t0,2)} secs")
    

def run(experiment:str="vivo_horvath", run:str="probe_1"):

    # SETUP CONFIG
    cfg, param_conf = get_config(experiment, run).values()

    # track time
    t0 = time.time()
    if CAST:
        cast_as_RecordingExtractor(cfg)
    if SAVE_METADATA:
        Wired = save_metadata(experiment, run, cfg)
    if PREPROCESS:
        #Wired = load_wired(cfg)
        preprocess_recording(Wired, cfg, param_conf, job_dict)
    logger.info(f"All completed in {np.round(time.time()-t0,2)} secs")