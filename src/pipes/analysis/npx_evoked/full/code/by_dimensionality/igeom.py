"""Pipeline that measures the information capacity
of ground truth unit population for equiprobable sampling
for different dimensionality. We reduce the dimensionality 
of the N x D unit population response to N_NEW x D for all 
conditions to enable comparison between conditions with different 
number of units.

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/analysis/npx_evoked/full/code/by_dimensionality/igeom.sh

8 directions :
5h for 200 bootstraps
1h34 for 20 bootstraps
9 mins for 5 bootstraps
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
SEED = 0
BLOCK = 0
N_BOOT = 200               
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
IGEOM = data_conf["analyses"]["neural_code"]["by_dimensionality"]["igeom"]

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
    "stimulus_intervals_ms": stimulus_intervals_ms,
    "quality_path": quality_path,
    "block": BLOCK,
    "n_boot": N_BOOT,
}
igeom_params = {
    "stimulus_labels": stimulus_labels,
    "sample_classes": SAMPLE_DIR,
    "n_exple_per_class": n_exple_per_class,
    "seed": SEED,
    "reduce_dim": True   # else sometimes crashes
}

# load sorted unit quality table  
quality_df = pd.read_csv(quality_path)


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
    df_0 = None
    df_1 = None
    df_2 = None
    df_3 = None
    df_4 = None
    df_5 = None
    
    # get the entire ground truth unit population
    if rank == 0:
        
        igeom_params["n_new"] = 50
        
        logger.info("Computing igeom for entire ground truth unit population.")
        df = igeom.get_igeom_metrics_bootstrapped_for_ground_truth2(stimulus_intervals_ms,
                                                    GT,
                                                    params=igeom_params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)
        df["Sampling scheme"] = "entire ground truth"
        
        logger.info("Computing igeom for randomly sampled ground truth unit population.")
        rand_df = igeom.get_igeom_stats_for_gt_and_random_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, **params)
        
        logger.info("Computing igeom for biased sampled ground truth unit population.")
        bias_df = igeom.get_igeom_stats_for_gt_and_biased_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, dt=DT, **params)        
        
        # concatenate
        df_0 = pd.concat([df, rand_df, bias_df])
        df_0["dimensionality"] = igeom_params["n_new"]
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")


    # sub-sample ground truth to match KS4 sampling bias
    if rank == 1:
        
        igeom_params["n_new"] = 100
        
        logger.info("Computing igeom for entire ground truth unit population.")
        df = igeom.get_igeom_metrics_bootstrapped_for_ground_truth2(stimulus_intervals_ms,
                                                    GT,
                                                    params=igeom_params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)

        logger.info("Computing igeom for randomly sampled ground truth unit population.")
        rand_df = igeom.get_igeom_stats_for_gt_and_random_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, **params)
        
        logger.info("Computing igeom for biased sampled ground truth unit population.")
        bias_df = igeom.get_igeom_stats_for_gt_and_biased_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, dt=DT, **params)        
        
        # concatenate
        df_1 = pd.concat([df, rand_df, bias_df])
        df_1["dimensionality"] = igeom_params["n_new"]
        
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")
    
            
    # sub-sample ground truth to match KS3 sampling bias
    elif rank == 2:
        
        igeom_params["n_new"] = 200
        
        logger.info("Computing igeom for entire ground truth unit population.")
        df = igeom.get_igeom_metrics_bootstrapped_for_ground_truth2(stimulus_intervals_ms,
                                                    GT,
                                                    params=igeom_params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)
        
        logger.info("Computing igeom for randomly sampled ground truth unit population.")
        rand_df = igeom.get_igeom_stats_for_gt_and_random_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, **params)
        
        logger.info("Computing igeom for biased sampled ground truth unit population.")
        bias_df = igeom.get_igeom_stats_for_gt_and_biased_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, dt=DT, **params)        
        
        # concatenate
        df_2 = pd.concat([df, rand_df, bias_df])        
        df_2["dimensionality"] = igeom_params["n_new"]
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # biased sampling of ground truth that matches spike sorting output
    elif rank == 3:
        
        igeom_params["n_new"] = 400
        
        logger.info("Computing igeom for entire ground truth unit population.")
        df = igeom.get_igeom_metrics_bootstrapped_for_ground_truth2(stimulus_intervals_ms,
                                                    GT,
                                                    params=igeom_params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)
        
        logger.info("Computing igeom for randomly sampled ground truth unit population.")
        rand_df = igeom.get_igeom_stats_for_gt_and_random_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, **params)
        
        logger.info("Computing igeom for biased sampled ground truth unit population.")
        bias_df = igeom.get_igeom_stats_for_gt_and_biased_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, dt=DT, **params)        
        
        # concatenate
        df_3 = pd.concat([df, rand_df, bias_df])
        df_3["dimensionality"] = igeom_params["n_new"]        
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # biased sampling of ground truth that matches spike sorting output
    elif rank == 4:
        
        igeom_params["n_new"] = 800
        
        logger.info("Computing igeom for entire ground truth unit population.")
        df = igeom.get_igeom_metrics_bootstrapped_for_ground_truth2(stimulus_intervals_ms,
                                                    GT,
                                                    params=igeom_params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)

        logger.info("Computing igeom for randomly sampled ground truth unit population.")
        rand_df = igeom.get_igeom_stats_for_gt_and_random_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, **params)
        
        logger.info("Computing igeom for biased sampled ground truth unit population.")
        bias_df = igeom.get_igeom_stats_for_gt_and_biased_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, dt=DT, **params)        
        
        # concatenate
        df_4 = pd.concat([df, rand_df, bias_df])
        df_4["dimensionality"] = igeom_params["n_new"]
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # biased sampling of ground truth that matches spike sorting output
    elif rank == 5:
        
        igeom_params["n_new"] = 1600
        
        logger.info("Computing igeom for entire ground truth unit population.")
        df = igeom.get_igeom_metrics_bootstrapped_for_ground_truth2(stimulus_intervals_ms,
                                                    GT,
                                                    params=igeom_params,
                                                    sample_size=None,
                                                    block=BLOCK,
                                                    n_boot=N_BOOT)

        logger.info("Computing igeom for randomly sampled ground truth unit population.")
        rand_df = igeom.get_igeom_stats_for_gt_and_random_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, **params)
        
        logger.info("Computing igeom for biased sampled ground truth unit population.")
        bias_df = igeom.get_igeom_stats_for_gt_and_biased_sampling(sorter="KS4", sorting_path=K4, sorting_true_path=GT, params=igeom_params, dt=DT, **params)        
        
        # concatenate
        df_5 = pd.concat([df, rand_df, bias_df])
        df_5["dimensionality"] = igeom_params["n_new"]
        logger.info(f"Done getting info-geom metrics on rank {rank} in {np.round(time.time()-t0,2)} secs")

    logger.info(f"Completed on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # gather data on the master node 0
    # note: comm.gather() collects the data in a list such that the data
    # from the node 0 is at entry 0, from the node 1 at 1, node 2 at 2...
    # in the list
    df_1 = comm.gather(df_1, root=0)
    df_2 = comm.gather(df_2, root=0)
    df_3 = comm.gather(df_3, root=0)
    df_4 = comm.gather(df_4, root=0)
    df_5 = comm.gather(df_5, root=0)
    logger.info(f"Data gathered on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # concatenate all on node 0
    if rank == 0:
        
        # we re-convert the produced array
        # into dataframe
        df_0 = pd.DataFrame(df_0)
        df_1 = pd.DataFrame(df_1[1])
        df_2 = pd.DataFrame(df_2[2])
        df_3 = pd.DataFrame(df_3[3])
        df_4 = pd.DataFrame(df_4[4])
        df_5 = pd.DataFrame(df_5[5])
        
        # get from node rank 1, 2 ..
        # node that produced that data 1, 2s
        # and sort by quality
        df = pd.concat([df_0, df_1, df_2, df_3, df_4, df_5])
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