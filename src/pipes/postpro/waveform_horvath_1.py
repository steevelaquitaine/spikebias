"""pipeline to compute Waveform Extractor for horvath et al.'s  recording depth 1

author: steeve.laquitaine@epfl.ch
date: 19.10.2023
modified: 19.10.2023

Usage:
    
    # submit to cluster
    sbatch cluster/postpro/extract_waveform_horvath_1.sbatch
"""
# SETUP PACKAGES
import os
import logging
import logging.config
import yaml
import shutil
from time import time
import spikeinterface as si
from spikeinterface.comparison import GroundTruthStudy
from src.nodes.prepro import preprocess

# SET PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/sfn_2023"
os.chdir(PROJ_PATH)
from src.nodes.utils import get_config

# SET PARAMETERS
EXPERIMENT = "vivo_horvath"   
SIMULATION_DATE = "2021_file_1"

# waveform period in ms
MS_BEFORE = 3  
MS_AFTER = 3

# SETUP CONFIG
data_conf, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()
STUDY_FOLDER = data_conf["postprocessing"]["waveform"]["study"]
REC_FOLDER_1 = data_conf["preprocessing"]["output"]["trace_file_path"]
SORTED_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def run():
    """extract spike waveforms from near-contact cells
    add a cell_id argument to filter by cell_ids

    takes 10 min

    Args:
        experiment (str):
        simulation_date (str): _description_
        lfp_trace_file (str): _description_
        spike_file (str): _description_
        cell_id (int): _description_
        study_folder (str): _description_
        ms_before (float): _description_
        ms_after (float): _description_

    Returns:
        _type_: waveform extractor
    """
    # get Sorting extractor
    Sorting = si.load_extractor(SORTED_PATH)
    logger.info(f"loaded sorting extractor in {time()-t0} secs")

    # get preprocessed traces
    trace = preprocess.load(data_conf)
    logger.info(f"loaded preprocessed recording extractor in {time()-t0} secs")

    # bundle (12 secs)
    t0 = time()
    gt_dict = {"rec0": (trace, Sorting)}
    shutil.rmtree(STUDY_FOLDER, ignore_errors=True)
    study = GroundTruthStudy.create(STUDY_FOLDER, gt_dict)
    logger.info(f"created study in {time()-t0} secs")
    
    # compute waveforms (0.21s, 312 spikes)
    # - this creates and write npy waveforms files on disk in study/waveforms/
    # - note: for a single cell, no gain from multiprocessing
    # - 100 jobs is slow
    t0 = time()
    study.compute_waveforms(trace, ms_before=MS_BEFORE, ms_after=MS_AFTER, n_jobs=-1, total_memory="1G")
    logger.info(f"computed waveforms in {time()-t0} secs")

    # create waveform extractor from saved folder
    we = study.get_waveform_extractor(trace)
    logger.info(f"returned waveforms in {time()-t0} secs")
    return we


if __name__ == "__main__":
    logger.info(f"Computing Waveform extractor for in vivo horvath file 1...")
    we = run()
    logger.info("All done")