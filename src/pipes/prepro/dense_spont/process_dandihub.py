"""pipeline for preprocessing the dense probe's simulated recordings

usage:

duration: 13 min on DANDI hub's "large" server.

Specs recommendations:
    - between 30 and 64 GB RAM is enough for 10 GB NWB datasets (free -g); this was tested on 750 GB RAM.
"""

import os
import logging
import logging.config
import yaml
import time 
import numpy as np

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

# spikebias package
from src.nodes.utils import get_config
from src.nodes.prepro import preprocess

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
    
    # load RecordingExtractor and Ground truth SortingExtractor
    Recording, Sorting = preprocess.load_on_dandihub(data_conf)


    # fit noise and gain
    if FIT_CAST:
        Recording = preprocess.fit_and_cast_as_extractor_dense_probe_on_dandihub(Recording,
                                                                                 data_conf=data_conf,
                                                                                 param_conf=param_conf,
                                                                                 offset=OFFSET,
                                                                                 scale_and_add_noise=SCALE_AND_ADD_NOISE)
    
    if WIRE:
        preprocess.wire_probe(data_conf,
                              param_conf,
                              Recording,
                              blueconfig=None,
                              save_metadata=SAVE_METADATA,
                              job_dict=job_dict,
                              n_sites=128,
                              load_atlas_metadata=True,
                              load_filtered_cells_metadata=True)
    
    if PREPROCESS:
        preprocess.preprocess_recording_dense_probe(data_conf=data_conf,
                                                    param_conf=param_conf,
                                                    job_dict=job_dict)
    
    if GROUND_TRUTH:
        preprocess.save_ground_truth_on_dandihub(Sorting, data_conf)

        
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")