"""Pipeline that measure the information capacity
of ground truth unit and sorted populations for each 
spike sorter.

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/analysis/npx_evoked/full/code/by_sorter/igeom.sh
    or 
    python3.9 -m src.pipes.analysis.npx_evoked.full.code.by_sorter.igeom

200 is the final used.
    * 200 boots -> 1h14
"""

import os 
import numpy as np
import pandas as pd
import yaml
import logging
import logging.config
import time 
from mpi4py import MPI

# set project path
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.analysis.code import igeom
from src.nodes import utils

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# PARAMETERS
DT = 1.3                  # ms (optimized)
THR_GOOD = 0.8            # threshold sorting accuracy for high-quality units ("good")
N_NEW = 200               # default: 200; nb of new neural latents resulting from Gaussian random projection (); low values fail
SEED = 0                  # reproducibility
BLOCK = 0
N_BOOT = 200         # 240 takes 6 hours
SAMPLE_DIR = np.arange(0, 360, 45) # sample 8 directions out of the 360; (4, 5, 6, fails, 8 worked directions)

# SETUP CONFIG
data_conf, _ = get_config("silico_neuropixels", "npx_evoked").values()
GT = data_conf["sorting"]["simulation"]["ground_truth"]["full"]["output"]
K4 = data_conf["sorting"]["sorters"]["kilosort4"]["full"]["output"]
K3 = data_conf["sorting"]["sorters"]["kilosort3"]["full"]["output"]
K25 = data_conf["sorting"]["sorters"]["kilosort2_5"]["full"]["output"]
K2 = data_conf["sorting"]["sorters"]["kilosort2"]["full"]["output"]
KS = data_conf["sorting"]["sorters"]["kilosort"]["full"]["output"]
HS = data_conf["sorting"]["sorters"]["herdingspikes"]["full"]["output"]
REC = data_conf["probe_wiring"]["output"]
IGEOM = data_conf["analyses"]["neural_code"]["by_sorter"]["igeom"]
QUALITY_PATH = data_conf["analyses"]["quality"]

# COMMON DATASETS
# get task epoch infos
task = igeom.get_task_parameters(
    start_delay=500,
    n_orientations=10,
    n_repeats=50,
    stimulus_duration=200,
    n_simulations=36,
)

# get stimulus directions and intervals (ms)
stimulus_labels = igeom.get_stimulus_labels()
stimulus_intervals_ms = igeom.get_stimulus_intervals_ms(
    task["epoch_labels"], task["epoch_ms"]
)

# number of sample instances per class
n_exple_per_class = sum(stimulus_labels == 0)

# parameters
igeom_params = {
    "stimulus_labels": stimulus_labels,
    "sample_classes": SAMPLE_DIR,
    "n_exple_per_class": n_exple_per_class,
    "n_new": N_NEW,
    "seed_shuffling": 0,
    "reduce_dim": True   # else sometimes crashes
}

# load sorted unit quality table  
quality_df = pd.read_csv(QUALITY_PATH)


def main(rank):
    """entry point

    Args:
        rank (_type_): _description_
    """

    # track time
    t_start = time.time()
    t0 = time.time()

    logger.info(f"Started computing sorted unit quality on rank {rank}")
    
    # pre-allocate the output of each compute node
    df_gt = None
    df_k4 = None
    df_k3 = None
    df_k25 = None
    df_k2 = None
    df_ks = None
    df_hs = None
    
    # get the entire ground truth unit population
    if rank == 0:

        logger.info("Computing igeom for entire ground truth unit population.")
        df_gt = igeom.get_igeom_metrics_bootstrapped_for_ground_truth(stimulus_intervals_ms,
                                                    GT,
                                                    params=igeom_params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")


    # sub-sample ground truth to match KS4 sampling bias
    if rank ==1:
        
        logger.info("Computing igeom for sorted unit population.")
        df_k4 = igeom.get_igeom_metrics_for_sorter_bootstrapped("KS4",
                                   K4,
                                   stimulus_intervals_ms,
                                   params=igeom_params,
                                   block=0,
                                   n_boot=N_BOOT)
            
    # sub-sample ground truth to match KS3 sampling bias
    elif rank == 2:
        
        logger.info("Computing igeom for sorted unit population.")
        df_k3 = igeom.get_igeom_metrics_for_sorter_bootstrapped("KS3",
                                   K3,
                                   stimulus_intervals_ms,
                                   params=igeom_params,
                                   block=0,
                                   n_boot=N_BOOT)
            

    # biased sampling of ground truth that matches spike sorting output
    elif rank == 3:
        
        logger.info("Computing igeom for sorted unit population.")
        df_k25 = igeom.get_igeom_metrics_for_sorter_bootstrapped("KS2.5",
                                   K25,
                                   stimulus_intervals_ms,
                                   params=igeom_params,
                                   block=0,
                                   n_boot=N_BOOT)
            
    # biased sampling of ground truth that matches spike sorting output
    elif rank == 4:
        
        logger.info("Computing igeom for sorted unit population.")
        df_k2 = igeom.get_igeom_metrics_for_sorter_bootstrapped("KS2",
                                   K2,
                                   stimulus_intervals_ms,
                                   params=igeom_params,
                                   block=0,
                                   n_boot=N_BOOT)
    # biased sampling of ground truth that matches spike sorting output
    elif rank == 5:
        
        logger.info("Computing igeom for sorted unit population.")
        df_ks = igeom.get_igeom_metrics_for_sorter_bootstrapped("KS",
                                   KS,
                                   stimulus_intervals_ms,
                                   params=igeom_params,
                                   block=0,
                                   n_boot=N_BOOT)

    # biased sampling of ground truth that matches spike sorting output
    elif rank == 6:
        
        logger.info("Computing igeom for sorted unit population.")
        df_hs = igeom.get_igeom_metrics_for_sorter_bootstrapped("HS",
                                   HS,
                                   stimulus_intervals_ms,
                                   params=igeom_params,
                                   block=0,
                                   n_boot=N_BOOT)
        
    logger.info(f"Completed on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # gather data on the master node 0
    # note: comm.gather() collects the data in a list such that the data
    # from the node 0 is at entry 0, from the node 1 at 1, node 2 at 2...
    # in the list
    df_k4 = comm.gather(df_k4, root=0)
    df_k3 = comm.gather(df_k3, root=0)
    df_k25 = comm.gather(df_k25, root=0)
    df_k2 = comm.gather(df_k2, root=0)
    df_ks = comm.gather(df_ks, root=0)
    df_hs = comm.gather(df_hs, root=0)
    logger.info(f"Data gathered on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # concatenate all on node 0
    if rank == 0:
        
        # we re-convert the produced array
        # into dataframe
        df_gt = pd.DataFrame(df_gt)
        df_k4 = pd.DataFrame(df_k4[1])
        df_k3 = pd.DataFrame(df_k3[2])
        df_k25 = pd.DataFrame(df_k25[3])
        df_k2 = pd.DataFrame(df_k2[4])
        df_ks = pd.DataFrame(df_ks[5])
        df_hs = pd.DataFrame(df_hs[6])
        
        # get from node rank 1, 2 ..
        # node that produced that data 1, 2s
        # and sort by quality
        df = pd.concat([df_gt, df_k4, df_k3, df_k25, df_k2, df_ks, df_hs])
        logger.info(f"Concatenated dataframe on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # save
        utils.create_if_not_exists(os.path.dirname(IGEOM))
        df.to_csv(IGEOM, index=False)
        logger.info(f"Saved csv on {rank} in {np.round(time.time()-t0,2)} secs")

    logger.info(f"Done saving results at : {IGEOM}")
    logger.info(f"All completed in {np.round(time.time()-t_start,2)} secs")
        
# run
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
main(rank)