"""Pipeline that measure the information capacity
of the sorted unit population by unit quality class and 
spike sorter

block 0: 5 bootstraps

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/analysis/npx_evoked/full/code/by_sorter_and_q2/igeom1.sh
    or 
    python3.9 -m src.pipes.analysis.npx_evoked.full.code.by_sorter_and_q2.igeom1

takes 4 minutes for 5 bootstraps, 8 directions
takes 24 minutes for 30 bootstraps, 8 directions
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
proj_path = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(proj_path)

from src.nodes.utils import get_config
from src.nodes.analysis.code import igeom
from src.nodes import utils

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# PARAMETERS
# best parameters so far: nb_units: 10; sample_size=60; N_NEW=sample_size
DT = 1.3                  # ms (optimized)
THR_GOOD = 0.8            # threshold sorting accuracy for high-quality units ("good")
N_NEW = 200               # default: 200; nb of new neural latents resulting from Gaussian random projection ()
SEED = 0                  # reproducibility
BLOCK = 1
N_BOOT = 200              # 360 should take 6 hours
SAMPLE_DIR = np.arange(0, 360, 45) # select 8 directions out of the 360; (4, 5, 6, fails, 8 worked directions)

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
IGEOM = data_conf["analyses"]["neural_code"]["by_sorter_and_quality"][f"igeom{BLOCK}"]
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
params = {
    "stimulus_labels": stimulus_labels,
    "sample_classes": SAMPLE_DIR,
    "n_exple_per_class": n_exple_per_class,
    "n_new": N_NEW,
    "seed": 0,
    "reduce_dim": True # else sometimes crashes 
}

# parameters to calculate information geometrics
igeom_params = {
    "quality_path": QUALITY_PATH,
                "stimulus_intervals_ms": stimulus_intervals_ms,
                "params": params,
                "seed": SEED,
                "n_boot": N_BOOT,
                "block": BLOCK,
                "temp_path": IGEOM,
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
    
    # pre-allocate
    df_k4 = None
    df_k3 = None
    df_k25 = None
    df_k2 = None
    df_ks = None
    df_hs = None
    
    if rank == 0:
        
        # Ground truth 
        df_gt = igeom.get_igeom_metrics_bootstrapped_for_ground_truth(stimulus_intervals_ms,
                                                    GT,
                                                    params=params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")

    if rank == 1:
        
        # KS4
        df_k4 = igeom.get_igeom_metrics_bootstrapped_by_sorter_and_q_wo_sampling_bias("KS4", K4, GT, DT, **igeom_params)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
    elif rank == 2:
        
        # KS3
        df_k3 = igeom.get_igeom_metrics_bootstrapped_by_sorter_and_q_wo_sampling_bias("KS3", K3, GT, DT, **igeom_params)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")
                
    elif rank == 3:
        
        # KS2.5
        df_k25 = igeom.get_igeom_metrics_bootstrapped_by_sorter_and_q_wo_sampling_bias("KS2.5", K25, GT, DT, **igeom_params)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
    elif rank == 4:
        
        # KS2
        df_k2 = igeom.get_igeom_metrics_bootstrapped_by_sorter_and_q_wo_sampling_bias("KS2", K2, GT, DT, **igeom_params)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
    elif rank == 5:
        
        # KS
        df_ks = igeom.get_igeom_metrics_bootstrapped_by_sorter_and_q_wo_sampling_bias("KS", KS, GT, DT,**igeom_params)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
    elif rank == 6:
        
        # KS
        df_hs = igeom.get_igeom_metrics_bootstrapped_by_sorter_and_q_wo_sampling_bias("HS", HS,  GT, DT, **igeom_params)
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")
                
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