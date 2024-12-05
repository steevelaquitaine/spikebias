"""Pipeline that measure the information capacity
of the sorted unit population by unit quality class and 
spike sorter

block 0: 5 bootstraps

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/analysis/npx_evoked/full/code/by_quality/igeom.sh
    
takes 50 minutes for 5 bootstraps, 20 directions
takes 10 minutes for 5 bootstraps, 8 directions
"""

import os 
import numpy as np
import pandas as pd
import yaml
import logging
import logging.config
import time 
from mpi4py import MPI
from collections import defaultdict
import spikeinterface as si 

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
NB_UNITS = 10             # 10; minimum sample size to calculate information capacity
SAMPLE_SIZE = 2000         # 60; number of sampled units used to calculate information capacity for all conditions
N_NEW = 200                # default: 200; nb of new neural latents resulting from Gaussian random projection (); low values fail
SEED = 0                  # reproducibility
BLOCK = 0
N_BOOT = 20               # 240 takes 6 hours
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
IGEOM = data_conf["analyses"]["neural_code"]["by_quality"]["igeom"]

# sorted unit quality path
quality_path = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/dataeng/0_silico/4_spikesorting_stimulus_test_neuropixels_8-1-24__8slc_80f_360r_50t_200ms_1_smallest_fiber_gids/0fcb7709-b1e9-4d84-b056-5801f20d55af/analysis/sorting_quality/sorting_quality_1h.csv"

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
    "quality_path": quality_path,
                "stimulus_intervals_ms": stimulus_intervals_ms,
                "params": params,
                "nb_units": NB_UNITS,
                "sample_size": SAMPLE_SIZE,
                "seed": SEED,
                "n_boot": N_BOOT,
                "block": BLOCK,
                "temp_path": IGEOM,
                }

# load sorted unit quality table       
quality_df = pd.read_csv(quality_path)


def main():
    """entry point

    Args:
        rank (_type_): _description_
    """

    # track time
    t_start = time.time()

    # ground truth
    df_gt = igeom.get_igeom_metrics_bootstrapped_for_ground_truth(stimulus_intervals_ms,
                                                GT,
                                                params=params,
                                                sample_size=SAMPLE_SIZE,
                                                block=BLOCK,
                                                n_boot=N_BOOT)
        
    # spike sorters
    df = igeom.get_igeom_metrics_by_quality_bootstrapped(K4, K3, K25, K2, KS, HS, **igeom_params)

    # concatenate
    df = pd.concat([df_gt, df])
    
    # save
    utils.create_if_not_exists(os.path.dirname(IGEOM))
    df.to_csv(IGEOM, index=False)
    logger.info(f"Done saving results at : {IGEOM}")
    logger.info(f"All completed in {np.round(time.time()-t_start,2)} secs")
    
# run
main()