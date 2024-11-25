"""pipeline to process the "Buccino"'s experiment

  author: steeve.laquitaine@epfl.ch
    date: 13.12.2023
modified: 10.07.2024

 usage:

    sbatch cluster/prepro/buccino/process_gain_ftd.sh
           
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
import spikeinterface.extractors as se

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.prepro import preprocess
from src.nodes import utils

# SETUP LOGGING
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SETUP PARALLEL PROCESSING
# required, else fit and cast as extractor crashes due to lack of 
# memory
job_dict = {"n_jobs": 1, "chunk_memory": None, "progress_bar": True} # butterworth


SAVE_RECORDINGEXTRACTOR = False

def fit_gain(gain_adjust: float, cfg_b: dict, cfg_v: dict):
    
    # track time
    t0 = time.time()
    
    # get marques-smith config
    RAW_PATH_v = cfg_v["probe_wiring"]["full"]["output"]

    # get synth. model config
    RAW_PATH_b = cfg_b["probe_wiring"]["full"]["input"]
    FIT_PATH = cfg_b["preprocessing"]["fitting"]["fitted_gain"]
    TUNED_GAIN = cfg_b["preprocessing"]["fitting"]["tuned_gain"]
    
    # max absolute amplitude in layer 5 for raw vivo
    raw_vivo = si.load_extractor(RAW_PATH_v)
    raw_vivo = spre.astype(raw_vivo, "int16")
    l5 = raw_vivo.get_property("layers") == "L5"
    max_v = np.max(np.absolute(raw_vivo.get_traces(channel_ids=l5)))

    # max absolute amplitude in raw synthetic (layer 5)
    raw_bucci = si.load_extractor(RAW_PATH_b)
    SF_b = raw_bucci.get_sampling_frequency()
    raw_b_10m = raw_bucci.frame_slice(start_frame=0, end_frame=10 * 60 * SF_b)
    raw_b_10m = spre.astype(raw_b_10m, "int16")
    traces = raw_b_10m.get_traces()
    max_b = np.absolute(traces).max()
    
    # calculate gain differemce
    fitted_gain = max_v / max_b
    utils.create_if_not_exists(os.path.dirname(FIT_PATH))
    np.save(FIT_PATH, fitted_gain)
    
    # tune gain
    tuned_gain = fitted_gain * gain_adjust
    np.save(TUNED_GAIN, tuned_gain)    
    logger.info(f"Fitted and tuned gain in {np.round(time.time()-t0,2)} secs")
    
    
def fit_and_save(cfg: dict):
    """apply gain

    Args:
        cfg (dict): _description_

    Returns:
        _type_: _description_
    """
    # track time
    t0 = time.time()
    
    # get paths
    RAW_PATH_b = cfg["probe_wiring"]["full"]["input"]
    FITD_PATH = cfg["probe_wiring"]["full"]["output"]
    TUNED_GAIN = cfg["preprocessing"]["fitting"]["tuned_gain"]
    
    # load data
    Wired = si.load_extractor(RAW_PATH_b)
    gain = np.load(TUNED_GAIN).item()
    
    # apply gain
    Wired = spre.scale(Wired, gain=gain)
    
    # cast as int16
    Wired = spre.astype(Wired, "int16")
    
    # record "L5" metadata
    n_sites = Wired.get_num_channels()
    Wired.set_property("layers", ["L5"] * n_sites)
    
    # save recording
    shutil.rmtree(FITD_PATH, ignore_errors=True)
    Wired.save(
        folder=FITD_PATH,
        format="binary",
        **job_dict
    )
    logger.info(f"Applied gain and saved in {np.round(time.time()-t0,2)} secs")
    return Wired


def preprocess_recording(Wired, cfg, prm, job_dict: dict, filtering='butterworth'):
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
    Preprocessed = preprocess.run_butterworth_filtering_buccino(Wired, cfg,
                                  prm)
    # save
    shutil.rmtree(WRITE_PATH, ignore_errors=True)
    Preprocessed.save(folder=WRITE_PATH, format="binary", **job_dict)
    
    # check is preprocessed
    print(Preprocessed.is_filtered())
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")


def run(gain_adjust:float, filtering:str="wavelet"):
    """
    args:
        filtering: "butterworth" or "wavelet"
    """
    t0 = time.time()
    
    # Marques-Smith config
    cfg_v, _ = get_config("vivo_marques", "c26").values()
    
    # Synthetic model's config
    cfg_b, prm_b = get_config("buccino_2020", "2020").values()    
    
    # save RecordingExtractor
    if SAVE_RECORDINGEXTRACTOR:
        
        Recording = se.NwbRecordingExtractor(cfg_b["recording"]["input"])
        shutil.rmtree(cfg_b["recording"]["output"], ignore_errors=True)
        Recording.save(
            folder=cfg_b["recording"]["output"], format="binary", **job_dict
            )
    
    # fit gain
    fit_gain(gain_adjust, cfg_b, cfg_v)
    
    # apply gain and save wired recording
    Wired = fit_and_save(cfg_b)
    
    # preprocessing
    preprocess_recording(Wired, cfg_b, prm_b, job_dict, filtering)
    logger.info(f"Pipeline done in {np.round(time.time()-t0,2)} secs")