"""pipeline to extract sorted units' spike waveforms 
from silico marques, for near-contact cells

author: steeve.laquitaine@epfl.ch
date: 17.01.2023
modified: 17.01.2023

Usage:
    
    # submit to cluster
    sbatch cluster/postpro/get_waveforms.sh
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

# SET PROJECT PATH
from src.nodes.utils import get_config

# waveform period in ms
MS_BEFORE = 3
MS_AFTER = 3


# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def run(experiment: str, run: str, sorter: str, rec_path: str):
    """extract spike waveforms from near-contact cells
    add a cell_id argument to filter by cell_ids

    takes 10 min

    Args:
        experiment (str):
        run (str): _description_
        sorter (str): "kilosort3"
        rec_path (str): preprocessed recording path

    Returns:
        _type_: waveform extractor
    """
    logger.info("Started...")
    
    # SETUP CONFIG
    data_conf, _ = get_config(experiment, run).values()
    STUDY_FOLDER = data_conf["postprocessing"]["waveform"]["sorted"]["study"][sorter]
    SORTED_PATH = data_conf["sorting"]["sorters"][sorter]["output"]
    
    # get Sorting extractor
    Sorting = si.load_extractor(SORTED_PATH)

    # get preprocessed traces
    Recording = si.load_extractor(
        rec_path
    )

    # bundle (12 secs)
    t0 = time()
    gt_dict = {"rec0": (Recording, Sorting)}
    shutil.rmtree(STUDY_FOLDER, ignore_errors=True)
    study = GroundTruthStudy.create(STUDY_FOLDER, gt_dict)
    logger.info(f"Created study in {time()-t0} secs")
    
    # compute waveforms (0.21s, 312 spikes)
    # - this creates and write npy waveforms files on disk in study/waveforms/
    # - note: for a single cell, no gain from multiprocessing
    # - 100 jobs is slow
    t0 = time()
    study.compute_waveforms(Recording, ms_before=MS_BEFORE, ms_after=MS_AFTER, max_spikes_per_unit=500, n_jobs=-1, total_memory="8G")
    logger.info(f"Computed waveforms in {time()-t0} secs")

    # create waveform extractor from saved folder
    we = study.get_waveform_extractor(Recording)    
    logger.info("get_waveforms - done ")
    return we