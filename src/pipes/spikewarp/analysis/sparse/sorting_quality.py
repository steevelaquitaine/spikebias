"""Evaluate the quality of the sorted units by sorter and experiment
for the spikewarp evoked experiment (James Isbister's paper)

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/analysis/spikewarp/sparse/sorting_quality.sh
    
duration: 7 minutes
"""

import os 
import numpy as np
import pandas as pd
import yaml
import logging
import logging.config
import time 
from mpi4py import MPI
import spikeinterface as si

# set project path
proj_path = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(proj_path)

from src.nodes.utils import get_config
from src.nodes.metrics import quality
from src.nodes.utils import create_if_not_exists

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# PARAMETERS
DUR = 600 # 10 minutes recording
DT = 1.3 # ms (optimized)
THR_GOOD = 0.8

# DATASETS

# NPX
# Evoked (10m)
cfg_nb, _ = get_config("others/spikewarp", "2024_10_08").values()
GT_10m = cfg_nb["ground_truth"]["10m"]["output"]
KS4_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
KS3_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort3"]["10m"]["output"]
KS2_5_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort2_5"]["10m"]["output"]
KS2_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort2"]["10m"]["output"]
KS_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort"]["10m"]["output"]
HS_nb_10m = cfg_nb["sorting"]["sorters"]["herdingspikes"]["10m"]["output"]
REC_nb = cfg_nb["probe_wiring"]["full"]["output"]

# saved dataframe of sorted unit quality
quality_path = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/postpro/0_silico/17_spikesorting_stimulus_test_neuropixels_8-10-24__8slc_60f_72r_250t_200ms_0/a9bf068a-b940-4514-9e6c-6055283272cc/analysis/sorting_quality/sorting_quality.csv"


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
    df_ks4 = None
    df_ks3 = None
    df_ks2_5 = None
    df_ks2 = None
    df_ks = None
    df_hs = None
    
    # compute quality
    if rank == 0:
        
        # (8secs)get agreement scores
        # npx-synthetic
        scores_nb_ks4, Sorting_nb_ks4, SortingTrue_nb = quality.get_score_single_unit(
            KS4_nb_10m, GT_10m, DT
        )
        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # (8m)compute chance scores (parallelized)
        chance_nb_ks4 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks4, Sorting_nb_ks4, SortingTrue_nb, DUR, DT
        )
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # qualify all sorted single-units
        df_nb_ks4 = quality.qualify_sorted_units2(
            scores_nb_ks4.values, scores_nb_ks4.index.values, chance_nb_ks4.values, THR_GOOD, KS4_nb_10m
        )
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # set experiments metadata
        df_nb_ks4["sorter"] = "KS4"
        df_nb_ks4["experiment"] = "E"
        
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # concatenate
        df_ks4 = df_nb_ks4
                
    elif rank == 1:
        
        # KS3
        scores_nb_ks3, Sorting_nb_ks3, SortingTrue_nb = quality.get_score_single_unit(
            KS3_nb_10m, GT_10m, DT
        )
        logger.info(f"Score done on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks3 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks3, Sorting_nb_ks3, SortingTrue_nb, DUR, DT
        )
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")  
          
        # evaluate
        df_nb_ks3 = quality.qualify_sorted_units2(
            scores_nb_ks3.values, scores_nb_ks3.index.values, chance_nb_ks3.values, THR_GOOD, KS3_nb_10m
        )
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")  
        
        # set metadata
        df_nb_ks3["sorter"] = "KS3"
        df_nb_ks3["experiment"] = "E"
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")  
        
        # concatenate
        df_ks3 = df_nb_ks3
                
    elif rank == 2:
        
        # KS2.5
        scores_nb_ks2_5, Sorting_nb_ks2_5, SortingTrue_nb = quality.get_score_single_unit(
            KS2_5_nb_10m, GT_10m, DT
        )
        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")  
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks2_5 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks2_5, Sorting_nb_ks2_5, SortingTrue_nb, DUR, DT
        )
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # qualify all sorted single-units
        df_nb_ks2_5 = quality.qualify_sorted_units2(
            scores_nb_ks2_5.values,
            scores_nb_ks2_5.index.values,
            chance_nb_ks2_5.values,
            THR_GOOD,
            KS2_5_nb_10m
        )
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")     

        # set metadata
        df_nb_ks2_5["sorter"] = "KS2.5"
        df_nb_ks2_5["experiment"] = "E"
        logger.info(f"Metadata set on rank {rank} in {np.round(time.time()-t0,2)} secs")
                
        # concatenate
        df_ks2_5 = df_nb_ks2_5
                
    elif rank == 3:
        
        # KS2
        scores_nb_ks2, Sorting_nb_ks2, SortingTrue_nb = quality.get_score_single_unit(
            KS2_nb_10m, GT_10m, DT
        )
        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks2 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks2, Sorting_nb_ks2, SortingTrue_nb, DUR, DT
        )
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # qualify all sorted single-units
        df_nb_ks2 = quality.qualify_sorted_units2(
            scores_nb_ks2.values, scores_nb_ks2.index.values, chance_nb_ks2.values, THR_GOOD, KS2_nb_10m
        )
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")     

        # set metadata
        df_nb_ks2["sorter"] = "KS2"
        df_nb_ks2["experiment"] = "E"
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # concatenate
        df_ks2 = df_nb_ks2
        
    elif rank == 4:
        
        # KS
        scores_nb_ks, Sorting_nb_ks, SortingTrue_nb = quality.get_score_single_unit(
            KS_nb_10m, GT_10m, DT
        )
        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks = quality.precompute_chance_score(
            REC_nb, scores_nb_ks, Sorting_nb_ks, SortingTrue_nb, DUR, DT
        )
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # qualify all sorted single-units
        df_nb_ks = quality.qualify_sorted_units2(
            scores_nb_ks.values, scores_nb_ks.index.values, chance_nb_ks.values, THR_GOOD, KS_nb_10m
        )
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0, 2)} secs")

        # set metadata
        df_nb_ks["sorter"] = "KS"
        df_nb_ks["experiment"] = "E"
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")
                
        # concatenate
        df_ks = df_nb_ks
                
    elif rank == 5:
        
        # HS
        scores_nb_hs, Sorting_nb_hs, SortingTrue_nb = quality.get_score_single_unit(
            HS_nb_10m, GT_10m, DT
        )
        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # pre-compute chance scores (parallelized)
        chance_nb_hs = quality.precompute_chance_score(
            REC_nb, scores_nb_hs, Sorting_nb_hs, SortingTrue_nb, DUR, DT
        )
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # qualify all sorted single-units
        df_nb_hs = quality.qualify_sorted_units2(
            scores_nb_hs.values, scores_nb_hs.index.values, chance_nb_hs.values, THR_GOOD, HS_nb_10m
        )
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0, 2)} secs")

        # set metadata
        df_nb_hs["sorter"] = "HS"
        df_nb_hs["experiment"] = "E"
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0, 2)} secs")     
                
        # concatenate
        df_hs = df_nb_hs
        
    logger.info(f"Completed on rank {rank} in {np.round(time.time()-t0,2)} secs")

    # gather data on the master node 0
    # note: comm.gather() collects the data in a list such that the data
    # from the node 0 is at entry 0, from the node 1 at 1, node 2 at 2...
    # in the list
    df_ks3 = comm.gather(df_ks3, root=0)
    df_ks2_5 = comm.gather(df_ks2_5, root=0)
    df_ks2 = comm.gather(df_ks2, root=0)
    df_ks = comm.gather(df_ks, root=0)
    df_hs = comm.gather(df_hs, root=0)
    logger.info(f"Data gathered on rank {rank} in {np.round(time.time()-t0,2)} secs")
    
    # concatenate all on node 0
    if rank == 0:
        
        # check arrays
        # assert isinstance(df_ks4, np.ndarray)
        # assert isinstance(df_ks3, np.ndarray)
        df_ks4 = pd.DataFrame(df_ks4) # already on rank 0
        df_ks3 = pd.DataFrame(df_ks3[1])
        df_ks2_5 = pd.DataFrame(df_ks2_5[2])
        df_ks2 = pd.DataFrame(df_ks2[3])
        df_ks = pd.DataFrame(df_ks[4])
        df_hs = pd.DataFrame(df_hs[5])
        
        # get from node rank 1, 2 ..
        # node that produced that data 1, 2s
        # and sort by quality
        df = pd.concat([df_ks4, df_ks3, df_ks2_5, df_ks2, df_ks, df_hs])
        df = df.sort_values("quality")
        
        logger.info(f"Concatenated dataframe on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # save        
        create_if_not_exists(os.path.dirname(quality_path))
        df.to_csv(quality_path, index=False)
        logger.info(f"Saved csv on {rank} in {np.round(time.time()-t0,2)} secs")

    logger.info(f"Done saving results at : {quality_path}")
    logger.info(f"All completed in {np.round(time.time()-t_start,2)} secs")
    
# run
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
main(rank)