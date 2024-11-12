"""compute noise for neuropixels data (Marques-Smith data, the two biophysical models)

usage:
    
    sbatch cluster/validation/noise/mpi_npx_full.sh
    
duration: 16 mins
"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import os
import spikeinterface as si
import numpy as np
from mpi4py import MPI
import time 
import yaml
import logging
import logging.config

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes import utils
from src.nodes.utils import get_config
from src.nodes.validation import noise

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# vivo
cfg, _ = get_config("vivo_marques", "c26").values()
RAW_PATH_v = cfg["raw"]
PREP_PATH_v = cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_PATH_v = cfg["validation"]["noise"]["full"]

# silico best fitted gain and noise
cfg, _ = get_config("silico_neuropixels", "concatenated").values()
PREP_PATH_s = cfg["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]
NOISE_PATH_s = cfg["validation"]["noise"]["full"]

# Buccino with best fitted gain for layer 5
cfg, _ = get_config("buccino_2020", "2020").values()
PREP_PATH_b = cfg["preprocessing"]["output"]["trace_file_path_gain_ftd"]
NOISE_PATH_b = cfg["validation"]["noise"]["full"]

# neuropixels (evoked biophysical model)
cfg, _ = get_config("silico_neuropixels", "stimulus").values()
PREP_PATH_ne = cfg["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]
NOISE_PATH_ne = cfg["validation"]["noise"]["full"]


def main(rank):

    # track time
    t0 = time.time()
    
    logger.info(f"Started measuring noise on rank {rank}")
    
    # calculate and save noise
    if rank == 0:
        Rec = si.load_extractor(PREP_PATH_v)
        traces = Rec.get_traces()
        noise_ = noise.get_in_parallel_single_nv(traces)
        utils.write_npy(noise_, NOISE_PATH_v)

    if rank == 1:
        Rec = si.load_extractor(PREP_PATH_s)
        traces = Rec.get_traces()
        noise_ = noise.get_in_parallel_single_ns(traces)
        utils.write_npy(noise_, NOISE_PATH_s)

    if rank == 2:
        Rec = si.load_extractor(PREP_PATH_b)
        traces = Rec.get_traces()
        noise_ = noise.get_in_parallel_single_nb(traces)
        utils.write_npy(noise_, NOISE_PATH_b)

    if rank == 3:
        Rec = si.load_extractor(PREP_PATH_ne)
        traces = Rec.get_traces()
        noise_ = noise.get_in_parallel_single_ne(traces)
        utils.write_npy(noise_, NOISE_PATH_ne)     
        
    logger.info(f"Completed on rank {rank} in {np.round(time.time()-t0,2)} secs")
    logger.info("Noise data written.")
    
# run
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
main(rank)