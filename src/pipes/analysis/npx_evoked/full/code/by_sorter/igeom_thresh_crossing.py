"""Pipeline that measure the information capacity of ground truth unit and sorted populations for threshold crossing

author: steeve.laquitaine@epfl.ch

Usage:

    # 1. activate pca_manifold and 2. run:
    nohup python3.9 -m src.pipes.analysis.npx_evoked.full.code.by_sorter.igeom_thresh_crossing > out_thresh_cross_igeom.log

200 is the final used.
    * 200 boots -> 1h14

Execution time: 2442.33 secs (40 mins) for 200 boots, N_NEW=200, 8 directions sampled
"""
import os 
import numpy as np
import pandas as pd
import yaml
import logging
import logging.config
import time 

# set project path
PROJ_PATH = "/home/steeve/steeve/epfl/code/spikebias"
os.chdir(PROJ_PATH)

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
N_BOOT = 200             # 240 takes 6 hours
SAMPLE_DIR = np.arange(0, 360, 45) # sample 8 directions out of the 360; (4, 5, 6, fails, 8 worked directions)

# SETUP CONFIG
GT = "dataset/00_raw/ground_truth_npx_evoked"
TC = "temp/no_spike_sorting/sorting_peak_npx_evoked" # threshold crossing
IGEOM = "dataset/01_intermediate/realism/0_silico/4_spikesorting_stimulus_test_neuropixels_8-1-24__8slc_80f_360r_50t_200ms_1_smallest_fiber_gids/0fcb7709-b1e9-4d84-b056-5801f20d55af/analysis/neural_code/igeometrics_thresh_crossing.csv"

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

def main():
    """entry point
    """
    # track time
    t0 = time.time()

    # get the entire ground truth unit population
    df_gt = igeom.get_igeom_metrics_bootstrapped_for_ground_truth(stimulus_intervals_ms,
                                                GT,
                                                params=igeom_params,
                                                sample_size=None,
                                                block=BLOCK,
                                                n_boot=N_BOOT)

    # sub-sample ground truth to match KS4 sampling bias
    df_tc = igeom.get_igeom_metrics_for_thresh_crossing_bootstrapped(
                                TC,
                                stimulus_intervals_ms,
                                params=igeom_params,
                                block=0,
                                n_boot=N_BOOT)
            
    # we re-convert the produced array
    # into dataframe
    df_gt = pd.DataFrame(df_gt)
    df_tc = pd.DataFrame(df_tc)
    
    # get from node rank 1, 2 ..
    # node that produced that data 1, 2s
    # and sort by quality
    df = pd.concat([df_gt, df_tc])    

    # save
    utils.create_if_not_exists(os.path.dirname(IGEOM))
    df.to_csv(IGEOM, index=False)
    logger.info(f"Saved csv on in {np.round(time.time()-t0,2)} secs")

main()