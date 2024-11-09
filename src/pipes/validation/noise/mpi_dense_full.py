"""compute noise of dense probe recordings (Horvath setup, the biophysical model)
at the three depths

usage:

    sbatch cluster/validation/noise/mpi_dense_full.sh

duration: 6 mins
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
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.validation import noise

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# data ----------
# probe 1 
cfg, _ = get_config("vivo_horvath", "probe_1").values()
PREP_PATH_hv1 = cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_PATH_hv1 = cfg["analyses"]["traces"]["noise"]

# probe 2
cfg, _ = get_config("vivo_horvath", "probe_2").values()
PREP_PATH_hv2 = cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_PATH_hv2 = cfg["analyses"]["traces"]["noise"]

# probe 3
cfg, _ = get_config("vivo_horvath", "probe_3").values()
PREP_PATH_hv3 = cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_PATH_hv3 = cfg["analyses"]["traces"]["noise"]

# models ----------
# probe 1
cfg, _ = get_config("silico_horvath", "concatenated/probe_1").values()
PREP_PATH_hs1 = cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_PATH_hs1 = cfg["analyses"]["traces"]["noise"]

# probe 2
cfg, _ = get_config("silico_horvath", "concatenated/probe_2").values()
PREP_PATH_hs2 = cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_PATH_hs2 = cfg["analyses"]["traces"]["noise"]

# probe 3
cfg, _ = get_config("silico_horvath", "concatenated/probe_3").values()
PREP_PATH_hs3 = cfg["preprocessing"]["output"]["trace_file_path"]
NOISE_PATH_hs3 = cfg["analyses"]["traces"]["noise"]


def main(rank):

    # track time
    t0 = time.time()
    
    logger.info(f"Started measuring noise on rank {rank}")
    
    # calculate and save noise
    if rank == 0:
        Rec = si.load_extractor(PREP_PATH_hv1)
        traces = Rec.get_traces()
        layers = Rec.get_property("layers")
        noise_ = noise.get_noise_data_hv(traces, layers)
        noise_.to_csv(NOISE_PATH_hv1)

    if rank == 1:
        Rec = si.load_extractor(PREP_PATH_hv2)
        traces = Rec.get_traces()
        layers = Rec.get_property("layers")
        noise_ = noise.get_noise_data_hv(traces, layers)
        noise_.to_csv(NOISE_PATH_hv2)

    if rank == 2:
        Rec = si.load_extractor(PREP_PATH_hv3)
        traces = Rec.get_traces()
        layers = Rec.get_property("layers")
        noise_ = noise.get_noise_data_hv(traces, layers)
        noise_.to_csv(NOISE_PATH_hv3)
     
    if rank == 3:
        Rec = si.load_extractor(PREP_PATH_hs1)
        traces = Rec.get_traces()
        layers = Rec.get_property("layers")
        noise_ = noise.get_noise_data_hs(traces, layers)
        noise_.to_csv(NOISE_PATH_hs1)

    if rank == 4:
        Rec = si.load_extractor(PREP_PATH_hs2)
        traces = Rec.get_traces()
        layers = Rec.get_property("layers")
        noise_ = noise.get_noise_data_hs(traces, layers)
        noise_.to_csv(NOISE_PATH_hs2)

    if rank == 5:
        Rec = si.load_extractor(PREP_PATH_hs3)
        traces = Rec.get_traces()
        layers = Rec.get_property("layers")
        noise_ = noise.get_noise_data_hs(traces, layers)
        noise_.to_csv(NOISE_PATH_hs3)
                  
    logger.info(f"Completed on rank {rank} in {np.round(time.time()-t0,2)} secs")
    logger.info("Noise data written.")
    
# run
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
main(rank)