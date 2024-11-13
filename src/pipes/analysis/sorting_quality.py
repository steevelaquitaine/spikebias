"""Evaluate the quality of the sorted units by sorter and experiment

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/analysis/sorting_quality.sh
"""

import os 
import numpy as np
import pandas as pd
import yaml
import logging
import logging.config
import time 
from mpi4py import MPI

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.metrics import quality
from src.nodes.metrics.quality import get_scores_for_dense_probe as gscdp
from src.nodes.metrics.quality import get_chance_for_dense_probe as gchdp
from src.nodes.metrics.quality import combine_quality_across_dense_probe as cqadb

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
# Synthetic (10m)
cfg_nb, _ = get_config("buccino_2020", "2020").values()
GT_nb_10m = cfg_nb["ground_truth"]["10m"]["output"]
KS4_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
KS3_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort3"]["10m"]["output"]
KS2_5_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort2_5"]["10m"]["output"]
KS2_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort2"]["10m"]["output"]
KS_nb_10m = cfg_nb["sorting"]["sorters"]["kilosort"]["10m"]["output"]
HS_nb_10m = cfg_nb["sorting"]["sorters"]["herdingspikes"]["10m"]["output"]
REC_nb = cfg_nb["probe_wiring"]["full"]["output"]

# biophy spont (10m)
cfg_ns, _ = get_config("silico_neuropixels", "concatenated").values()
KS4_ns_10m = cfg_ns["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
KS3_ns_10m = cfg_ns["sorting"]["sorters"]["kilosort3"]["10m"]["output"]
KS2_5_ns_10m = cfg_ns["sorting"]["sorters"]["kilosort2_5"]["10m"]["output"]
KS2_ns_10m = cfg_ns["sorting"]["sorters"]["kilosort2"]["10m"]["output"]
KS_ns_10m = cfg_ns["sorting"]["sorters"]["kilosort"]["10m"]["output"]
HS_ns_10m = cfg_ns["sorting"]["sorters"]["herdingspikes"]["10m"]["output"]
GT_ns_10m = cfg_ns["ground_truth"]["10m"]["output"]
REC_ns = cfg_ns["probe_wiring"]["full"]["output"]

# biophy evoked
cfg_ne, _ = get_config("silico_neuropixels", "stimulus").values()
KS4_ne_10m = cfg_ne["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
KS3_ne_10m = cfg_ne["sorting"]["sorters"]["kilosort3"]["10m"]["output"]
KS2_5_ne_10m = cfg_ne["sorting"]["sorters"]["kilosort2_5"]["10m"]["output"]
KS2_ne_10m = cfg_ne["sorting"]["sorters"]["kilosort2"]["10m"]["output"]
KS_ne_10m = cfg_ne["sorting"]["sorters"]["kilosort"]["10m"]["output"]
HS_ne_10m = cfg_ne["sorting"]["sorters"]["herdingspikes"]["10m"]["output"]
GT_ne_10m = cfg_ne["ground_truth"]["10m"]["output"]
REC_ne = cfg_ne["probe_wiring"]["full"]["output"]

# DENSE PROBE 
# depth 1
cfg_ds1, _ = get_config("silico_horvath", "concatenated/probe_1").values()
K4_d1 = cfg_ds1["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
K3_d1 = cfg_ds1["sorting"]["sorters"]["kilosort3"]["10m"]["output"]
K25_d1 = cfg_ds1["sorting"]["sorters"]["kilosort2_5"]["10m"]["output"]
K2_d1 = cfg_ds1["sorting"]["sorters"]["kilosort2"]["10m"]["output"]
K_d1 = cfg_ds1["sorting"]["sorters"]["kilosort"]["10m"]["output"]
H_d1 = cfg_ds1["sorting"]["sorters"]["herdingspikes"]["10m"]["output"]
R_d1 = cfg_ds1["probe_wiring"]["full"]["output"]
T_d1 = cfg_ds1["ground_truth"]["10m"]["output"]

# depth 2
cfg_ds2, _ = get_config("silico_horvath", "concatenated/probe_2").values()
K4_d2 = cfg_ds2["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
K3_d2 = cfg_ds2["sorting"]["sorters"]["kilosort3"]["10m"]["output"]
K25_d2 = cfg_ds2["sorting"]["sorters"]["kilosort2_5"]["10m"]["output"]
K2_d2 = cfg_ds2["sorting"]["sorters"]["kilosort2"]["10m"]["output"]
K_d2 = cfg_ds2["sorting"]["sorters"]["kilosort"]["10m"]["output"]
H_d2 = cfg_ds2["sorting"]["sorters"]["herdingspikes"]["10m"]["output"]
R_d2 = cfg_ds2["probe_wiring"]["full"]["output"]
T_d2 = cfg_ds2["ground_truth"]["10m"]["output"]

# depth 3
cfg_ds3, _ = get_config("silico_horvath", "concatenated/probe_3").values()
K4_d3 = cfg_ds3["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
K3_d3 = cfg_ds3["sorting"]["sorters"]["kilosort3"]["10m"]["output"]
K25_d3 = cfg_ds3["sorting"]["sorters"]["kilosort2_5"]["10m"]["output"]
K2_d3 = cfg_ds3["sorting"]["sorters"]["kilosort2"]["10m"]["output"]
K_d3 = cfg_ds3["sorting"]["sorters"]["kilosort"]["10m"]["output"]
H_d3 = cfg_ds3["sorting"]["sorters"]["herdingspikes"]["10m"]["output"]
R_d3 = cfg_ds3["probe_wiring"]["full"]["output"]
T_d3 = cfg_ds3["ground_truth"]["10m"]["output"]


# saved dataframe of sorted unit quality
quality_path = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/analysis/sorting_quality/sorting_quality.csv"


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
        # KS4 
        
        # (8secs)get agreement scores
        # npx-synthetic
        scores_nb_ks4, Sorting_nb_ks4, SortingTrue_nb = quality.get_score_single_unit(
            KS4_nb_10m, GT_nb_10m, DT
        )
        # npx-biophy. spont
        scores_ns_ks4, Sorting_ns_ks4, SortingTrue_ns = quality.get_score_single_unit(
            KS4_ns_10m, GT_ns_10m, DT
        )
        # npx-biophy.evoked
        scores_ne_ks4, Sorting_ne_ks4, SortingTrue_ne = quality.get_score_single_unit(
            KS4_ne_10m, GT_ne_10m, DT
        )
        # dense-biophy.spont
        (k4_sc, k4_so, k4_t) = gscdp(K4_d1, K4_d2, K4_d3, T_d1, T_d2, T_d3, DT)
        
        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # (8m)compute chance scores (parallelized)
        chance_nb_ks4 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks4, Sorting_nb_ks4, SortingTrue_nb, DUR, DT
        )
        chance_ns_ks4 = quality.precompute_chance_score(
            REC_ns, scores_ns_ks4, Sorting_ns_ks4, SortingTrue_ns, DUR, DT
        )
        chance_ne_ks4 = quality.precompute_chance_score(
            REC_ne, scores_ne_ks4, Sorting_ne_ks4, SortingTrue_ne, DUR, DT
        )
        k4_ch = gchdp(DUR, DT, R_d1, R_d2, R_d3, **k4_sc, **k4_so, **k4_t)

        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # qualify all sorted single-units
        df_ns_ks4 = quality.qualify_sorted_units2(
            scores_ns_ks4.values, scores_ns_ks4.index.values, chance_ns_ks4.values, THR_GOOD, KS4_ns_10m
        )
        df_nb_ks4 = quality.qualify_sorted_units2(
            scores_nb_ks4.values, scores_nb_ks4.index.values, chance_nb_ks4.values, THR_GOOD, KS4_nb_10m
        )
        df_ne_ks4 = quality.qualify_sorted_units2(
            scores_ne_ks4.values, scores_ne_ks4.index.values, chance_ne_ks4.values, THR_GOOD, KS4_ne_10m
        )
        df_ds_ks4 = cqadb(THR_GOOD, K4_d1, K4_d2, K4_d3, **k4_sc, **k4_ch)

        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")

        # set experiments metadata
        df_ns_ks4["sorter"] = "KS4"
        df_ns_ks4["experiment"] = "NS"
        df_nb_ks4["sorter"] = "KS4"
        df_nb_ks4["experiment"] = "S"
        df_ne_ks4["sorter"] = "KS4"
        df_ne_ks4["experiment"] = "E"
        df_ds_ks4["sorter"] = "KS4"
        df_ds_ks4["experiment"] = "DS"
        
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # concatenate
        df_ks4 = pd.concat([df_ns_ks4, df_nb_ks4, df_ne_ks4, df_ds_ks4])
                
    elif rank == 1:
        # KS3
        scores_nb_ks3, Sorting_nb_ks3, SortingTrue_nb = quality.get_score_single_unit(
            KS3_nb_10m, GT_nb_10m, DT
        )
        scores_ns_ks3, Sorting_ns_ks3, SortingTrue_ns = quality.get_score_single_unit(
            KS3_ns_10m, GT_ns_10m, DT
        )
        scores_ne_ks3, Sorting_ne_ks3, SortingTrue_ne = quality.get_score_single_unit(
            KS3_ne_10m, GT_ne_10m, DT
        )
        (k3_sc, k3_so, k3_t) = gscdp(K3_d1, K3_d2, K3_d3, T_d1, T_d2, T_d3, DT)
        
        logger.info(f"Score done on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks3 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks3, Sorting_nb_ks3, SortingTrue_nb, DUR, DT
        )
        chance_ns_ks3 = quality.precompute_chance_score(
            REC_ns, scores_ns_ks3, Sorting_ns_ks3, SortingTrue_ns, DUR, DT
        )
        chance_ne_ks3 = quality.precompute_chance_score(
            REC_ne, scores_ne_ks3, Sorting_ne_ks3, SortingTrue_ne, DUR, DT
        )
        k3_ch = gchdp(DUR, DT, R_d1, R_d2, R_d3, **k3_sc, **k3_so, **k3_t)

        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")  
          
        # evaluate
        df_ns_ks3 = quality.qualify_sorted_units2(
            scores_ns_ks3.values, scores_ns_ks3.index.values, chance_ns_ks3.values, THR_GOOD, KS3_ns_10m
        )
        df_nb_ks3 = quality.qualify_sorted_units2(
            scores_nb_ks3.values, scores_nb_ks3.index.values, chance_nb_ks3.values, THR_GOOD, KS3_nb_10m
        )
        df_ne_ks3 = quality.qualify_sorted_units2(
            scores_ne_ks3.values, scores_ne_ks3.index.values, chance_ne_ks3.values, THR_GOOD, KS3_ne_10m
        )
        df_ds_ks3 = cqadb(THR_GOOD, K3_d1, K3_d2, K3_d3, **k3_sc, **k3_ch)

        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")  
        
        # set metadata
        df_ns_ks3["sorter"] = "KS3"
        df_ns_ks3["experiment"] = "NS"
        df_nb_ks3["sorter"] = "KS3"
        df_nb_ks3["experiment"] = "S"
        df_ne_ks3["sorter"] = "KS3"
        df_ne_ks3["experiment"] = "E"
        df_ds_ks3["sorter"] = "KS3"
        df_ds_ks3["experiment"] = "DS"
        
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")  
        
        # concatenate
        df_ks3 = pd.concat([df_ns_ks3, df_nb_ks3, df_ne_ks3, df_ds_ks3])
                
    elif rank == 2:
        # KS2.5
        scores_nb_ks2_5, Sorting_nb_ks2_5, SortingTrue_nb = quality.get_score_single_unit(
            KS2_5_nb_10m, GT_nb_10m, DT
        )
        scores_ns_ks2_5, Sorting_ns_ks2_5, SortingTrue_ns = quality.get_score_single_unit(
            KS2_5_ns_10m, GT_ns_10m, DT
        )
        scores_ne_ks2_5, Sorting_ne_ks2_5, SortingTrue_ne = quality.get_score_single_unit(
            KS2_5_ne_10m, GT_ne_10m, DT
        )
        (k25_sc, k25_so, k25_t) = gscdp(K25_d1, K25_d2, K25_d3, T_d1, T_d2, T_d3, DT)

        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")  
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks2_5 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks2_5, Sorting_nb_ks2_5, SortingTrue_nb, DUR, DT
        )
        chance_ns_ks2_5 = quality.precompute_chance_score(
            REC_ns, scores_ns_ks2_5, Sorting_ns_ks2_5, SortingTrue_ns, DUR, DT
        )
        chance_ne_ks2_5 = quality.precompute_chance_score(
            REC_ne, scores_ne_ks2_5, Sorting_ne_ks2_5, SortingTrue_ne, DUR, DT
        )
        k25_ch = gchdp(DUR, DT, R_d1, R_d2, R_d3, **k25_sc, **k25_so, **k25_t)

        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # qualify all sorted single-units
        df_ns_ks2_5 = quality.qualify_sorted_units2(
            scores_ns_ks2_5.values,
            scores_ns_ks2_5.index.values,
            chance_ns_ks2_5.values,
            THR_GOOD, KS2_5_ns_10m
        )
        df_nb_ks2_5 = quality.qualify_sorted_units2(
            scores_nb_ks2_5.values,
            scores_nb_ks2_5.index.values,
            chance_nb_ks2_5.values,
            THR_GOOD,
            KS2_5_nb_10m
        )
        df_ne_ks2_5 = quality.qualify_sorted_units2(
            scores_ne_ks2_5.values,
            scores_ne_ks2_5.index.values,
            chance_ne_ks2_5.values,
            THR_GOOD,
            KS2_5_ne_10m
        )
        df_ds_ks25 = cqadb(THR_GOOD, K25_d1, K25_d2, K25_d3, **k25_sc, **k25_ch)
        
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")     

        # set metadata
        df_ns_ks2_5["sorter"] = "KS2.5"
        df_ns_ks2_5["experiment"] = "NS"
        df_nb_ks2_5["sorter"] = "KS2.5"
        df_nb_ks2_5["experiment"] = "S"
        df_ne_ks2_5["sorter"] = "KS2.5"
        df_ne_ks2_5["experiment"] = "E"
        df_ds_ks25["sorter"] = "KS2.5"
        df_ds_ks25["experiment"] = "DS"
        
        logger.info(f"Metadata set on rank {rank} in {np.round(time.time()-t0,2)} secs")
                
        # concatenate
        df_ks2_5 = pd.concat([df_ns_ks2_5, df_nb_ks2_5, df_ne_ks2_5, df_ds_ks25])
                
    elif rank == 3:
        # KS2
        scores_nb_ks2, Sorting_nb_ks2, SortingTrue_nb = quality.get_score_single_unit(
            KS2_nb_10m, GT_nb_10m, DT
        )
        scores_ns_ks2, Sorting_ns_ks2, SortingTrue_ns = quality.get_score_single_unit(
            KS2_ns_10m, GT_ns_10m, DT
        )
        scores_ne_ks2, Sorting_ne_ks2, SortingTrue_ne = quality.get_score_single_unit(
            KS2_ne_10m, GT_ne_10m, DT
        )
        (k2_sc, k2_so, k2_t) = gscdp(K2_d1, K2_d2, K2_d3, T_d1, T_d2, T_d3, DT)

        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks2 = quality.precompute_chance_score(
            REC_nb, scores_nb_ks2, Sorting_nb_ks2, SortingTrue_nb, DUR, DT
        )
        chance_ns_ks2 = quality.precompute_chance_score(
            REC_ns, scores_ns_ks2, Sorting_ns_ks2, SortingTrue_ns, DUR, DT
        )
        chance_ne_ks2 = quality.precompute_chance_score(
            REC_ne, scores_ne_ks2, Sorting_ne_ks2, SortingTrue_ne, DUR, DT
        )
        k2_ch = gchdp(DUR, DT, R_d1, R_d2, R_d3, **k2_sc, **k2_so, **k2_t)

        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # qualify all sorted single-units
        df_ns_ks2 = quality.qualify_sorted_units2(
            scores_ns_ks2.values, scores_ns_ks2.index.values, chance_ns_ks2.values, THR_GOOD, KS2_ns_10m
        )
        df_nb_ks2 = quality.qualify_sorted_units2(
            scores_nb_ks2.values, scores_nb_ks2.index.values, chance_nb_ks2.values, THR_GOOD, KS2_nb_10m
        )
        df_ne_ks2 = quality.qualify_sorted_units2(
            scores_ne_ks2.values, scores_ne_ks2.index.values, chance_ne_ks2.values, THR_GOOD, KS2_ne_10m
        )
        df_ds_ks2 = cqadb(THR_GOOD, K2_d1, K2_d2, K2_d3, **k2_sc, **k2_ch)
        
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0,2)} secs")     

        # set metadata
        df_ns_ks2["sorter"] = "KS2"
        df_ns_ks2["experiment"] = "NS"
        df_nb_ks2["sorter"] = "KS2"
        df_nb_ks2["experiment"] = "S"
        df_ne_ks2["sorter"] = "KS2"
        df_ne_ks2["experiment"] = "E"
        df_ds_ks2["sorter"] = "KS2"
        df_ds_ks2["experiment"] = "DS"
        
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # concatenate
        df_ks2 = pd.concat([df_ns_ks2, df_nb_ks2, df_ne_ks2, df_ds_ks2])
        
    elif rank == 4:
        # KS
        scores_nb_ks, Sorting_nb_ks, SortingTrue_nb = quality.get_score_single_unit(
            KS_nb_10m, GT_nb_10m, DT
        )
        scores_ns_ks, Sorting_ns_ks, SortingTrue_ns = quality.get_score_single_unit(
            KS_ns_10m, GT_ns_10m, DT
        )
        scores_ne_ks, Sorting_ne_ks, SortingTrue_ne = quality.get_score_single_unit(
            KS_ne_10m, GT_ne_10m, DT
        )
        (K_sc, K_so, K_t) = gscdp(K_d1, K_d2, K_d3, T_d1, T_d2, T_d3, DT)

        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # pre-compute chance scores (parallelized)
        chance_nb_ks = quality.precompute_chance_score(
            REC_nb, scores_nb_ks, Sorting_nb_ks, SortingTrue_nb, DUR, DT
        )
        chance_ns_ks = quality.precompute_chance_score(
            REC_ns, scores_ns_ks, Sorting_ns_ks, SortingTrue_ns, DUR, DT
        )
        chance_ne_ks = quality.precompute_chance_score(
            REC_ne, scores_ne_ks, Sorting_ne_ks, SortingTrue_ne, DUR, DT
        )
        K_ch = gchdp(DUR, DT, R_d1, R_d2, R_d3, **K_sc, **K_so, **K_t)
   
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # qualify all sorted single-units
        df_ns_ks = quality.qualify_sorted_units2(
            scores_ns_ks.values, scores_ns_ks.index.values, chance_ns_ks.values, THR_GOOD, KS_ns_10m
        )
        df_nb_ks = quality.qualify_sorted_units2(
            scores_nb_ks.values, scores_nb_ks.index.values, chance_nb_ks.values, THR_GOOD, KS_nb_10m
        )
        df_ne_ks = quality.qualify_sorted_units2(
            scores_ne_ks.values, scores_ne_ks.index.values, chance_ne_ks.values, THR_GOOD, KS_ne_10m
        )
        df_ds_ks = cqadb(THR_GOOD, K_d1, K_d2, K_d3, **K_sc, **K_ch)

        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0, 2)} secs")

        # set metadata
        df_ns_ks["sorter"] = "KS"
        df_ns_ks["experiment"] = "NS"
        df_nb_ks["sorter"] = "KS"
        df_nb_ks["experiment"] = "S"
        df_ne_ks["sorter"] = "KS"
        df_ne_ks["experiment"] = "E"
        df_ds_ks["sorter"] = "KS"
        df_ds_ks["experiment"] = "DS"
        
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0,2)} secs")
                
        # concatenate
        df_ks = pd.concat([df_ns_ks, df_nb_ks, df_ne_ks, df_ds_ks])
                
    elif rank == 5:
        
        # HS
        scores_nb_hs, Sorting_nb_hs, SortingTrue_nb = quality.get_score_single_unit(
            HS_nb_10m, GT_nb_10m, DT
        )
        scores_ns_hs, Sorting_ns_hs, SortingTrue_ns = quality.get_score_single_unit(
            HS_ns_10m, GT_ns_10m, DT
        )
        scores_ne_hs, Sorting_ne_hs, SortingTrue_ne = quality.get_score_single_unit(
            HS_ne_10m, GT_ne_10m, DT
        )
        (H_sc, H_so, H_t) = gscdp(H_d1, H_d2, H_d3, T_d1, T_d2, T_d3, DT)        
        
        logger.info(f"Scores done on rank {rank} in {np.round(time.time()-t0,2)} secs")     
        
        # pre-compute chance scores (parallelized)
        chance_nb_hs = quality.precompute_chance_score(
            REC_nb, scores_nb_hs, Sorting_nb_hs, SortingTrue_nb, DUR, DT
        )
        chance_ns_hs = quality.precompute_chance_score(
            REC_ns, scores_ns_hs, Sorting_ns_hs, SortingTrue_ns, DUR, DT
        )
        chance_ne_hs = quality.precompute_chance_score(
            REC_ne, scores_ne_hs, Sorting_ne_hs, SortingTrue_ne, DUR, DT
        )
        H_ch = gchdp(DUR, DT, R_d1, R_d2, R_d3, **H_sc, **H_so, **H_t)
        
        logger.info(f"Chance done on rank {rank} in {np.round(time.time()-t0,2)} secs")
        
        # qualify all sorted single-units
        df_ns_hs = quality.qualify_sorted_units2(
            scores_ns_hs.values, scores_ns_hs.index.values, chance_ns_hs.values, THR_GOOD, HS_ns_10m
        )
        df_nb_hs = quality.qualify_sorted_units2(
            scores_nb_hs.values, scores_nb_hs.index.values, chance_nb_hs.values, THR_GOOD, HS_nb_10m
        )
        df_ne_hs = quality.qualify_sorted_units2(
            scores_ne_hs.values, scores_ne_hs.index.values, chance_ne_hs.values, THR_GOOD, HS_ne_10m
        )
        df_ds_h = cqadb(THR_GOOD, H_d1, H_d2, H_d3, **H_sc, **H_ch)
        
        logger.info(f"Quality done on rank {rank} in {np.round(time.time()-t0, 2)} secs")

        # set metadata
        df_ns_hs["sorter"] = "HS"
        df_ns_hs["experiment"] = "NS"
        df_nb_hs["sorter"] = "HS"
        df_nb_hs["experiment"] = "S"
        df_ne_hs["sorter"] = "HS"
        df_ne_hs["experiment"] = "E"
        df_ds_h["sorter"] = "HS"
        df_ds_h["experiment"] = "DS"
        
        logger.info(f"Metadata added on rank {rank} in {np.round(time.time()-t0, 2)} secs")     
                
        # concatenate
        df_hs = pd.concat([df_ns_hs, df_nb_hs, df_ne_hs, df_ds_h])
        
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
        df_ks4 = pd.DataFrame(df_ks4)
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
        df.to_csv(quality_path, index=False)
        logger.info(f"Saved csv on {rank} in {np.round(time.time()-t0,2)} secs")

    logger.info(f"Done saving results at : {quality_path}")
    logger.info(f"All completed in {np.round(time.time()-t_start,2)} secs")
    
# run
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
main(rank)