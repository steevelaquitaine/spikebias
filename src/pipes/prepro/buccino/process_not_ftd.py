"""pipeline to process the "silico_neuropixels" experiment - run "concatenated" 
(simulation of Marques probe) from raw bbp_workflow simulation files to ready-to-sort 
SpikeInterface's Recording and Sorting Extractors

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023
modified: 10.07.2024

 usage:

    sbatch cluster/prepro/buccino/not_ftd.sbatch
           
Note:
    - if preprocessing write crashes because of memory issue. Rerun with all pipeline nodes
    set to False except PREPROCESS=True
    - the trace array requires 240 GB RAM (free RAM is typically 636 GB on a compute core)

Duration: 24 mins

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
import spikeinterface.preprocessing as spre

# move to project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.prepro import preprocess

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PARALLEL PROCESSING
# required, else fit and cast as extractor crashes due to lack of 
# memory
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True} # butterworth

    
def wire(cfg: dict):
    """apply gain

    Args:
        cfg (dict): _description_

    Returns:
        _type_: _description_
    """
    # track time
    t0 = time.time()
    
    # get paths
    RAW_PATH_b = cfg["probe_wiring"]["output"]
    WRITE_PATH = cfg["probe_wiring"]["10m"]["output_gain_not_ftd_int16"]
    
    # load data
    Wired = si.load_extractor(RAW_PATH_b)
    
    # cast as int16
    Wired = spre.astype(Wired, "int16")
    
    # save recording
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Wired.save(
        folder=WRITE_PATH,
        format="binary",
        **job_dict
    )
    logger.info(f"Wired raw (not fitted) Recording in {np.round(time.time()-t0,2)} secs")
    return Wired


def preprocess_recording(Wired, cfg, prm, job_dict: dict, filtering='butterworth'):
    """preprocess recording and write

    Args:   
        job_dict
        filtering: 'butterworth' or 'wavelet'

    takes 15 min (vs. 32 min w/o multiprocessing)
    """
    # time track
    t0 = time.time()
    logger.info("Starting 'preprocess_recording'")

    # write path
    WRITE_PATH = cfg["preprocessing"]["output"]["trace_file_path_not_ftd"]
    
    # preprocess
    Preprocessed = preprocess.run_butterworth_filtering_buccino(Wired, cfg,
                                  prm)
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(filtering:str="wavelet"):
    """
    args:
        filtering: "butterworth" or "wavelet"
    """
    # track time
    t0 = time.time()
    
    # Synthetic model's config
    cfg_b, prm_b = get_config("buccino_2020", "2020").values()
    
    # apply gain and save wired recording
    Wired = wire(cfg_b)
    
    # preprocessing
    preprocess_recording(Wired, cfg_b, prm_b, job_dict, filtering)
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")