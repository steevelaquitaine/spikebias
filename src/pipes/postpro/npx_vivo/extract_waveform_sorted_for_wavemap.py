"""pipeline to compute Waveform Extractor for in vivo Marques-Smith et al.
to be used for wavemap clustering. We need longer spike length to align them.

author: steeve.laquitaine@epfl.ch
date: 08.02.2023
modified: 08.02.2023

Usage:
    
    sbatch cluster/postpro/marques_vivo/extract_waveform_sorted_for_wavemap.sbatch

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

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config

# waveform period in ms
MS_BEFORE = 6
MS_AFTER = 6

# SETUP CONFIG
data_conf, param_conf = get_config("vivo_marques", "c26").values()
SORTED_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]
STUDY_FOLDER = data_conf["postprocessing"]["waveform"]["sorted"]["for_wavemap"]["study"]

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def run():
    """extract spike waveforms from near-contact cells
    add a cell_id argument to filter by cell_ids

    takes 10 min

    Returns:
        _type_: waveform extractor
    """
    # track time
    t0 = time()
    logger.info(f"Starting extracting Waveform for in vivo Marques ...")

    # get Sorting extractor
    Sorting = si.load_extractor(SORTED_PATH)
    logger.info(f"loaded sorting extractor in {time()-t0} secs")

    # get preprocessed traces
    trace = preprocess.load(data_conf)
    logger.info(f"loaded preprocessed recording extractor in {time()-t0} secs")

    # bundle into a study object (12 secs)
    t0 = time()
    gt_dict = {"rec0": (trace, Sorting)}
    shutil.rmtree(STUDY_FOLDER, ignore_errors=True)
    study = GroundTruthStudy.create(STUDY_FOLDER, gt_dict)
    logger.info(f"bundled into a study object in {time()-t0} secs")
    
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


# run pipeline
we = run()
logger.info("All done")