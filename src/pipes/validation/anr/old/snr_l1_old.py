"""Compute and save peak amplitude-to-noise ratio in layer 1

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/validation/main/snr/snr_l1.sh
    
STATUS: does not work due to message passing memory limitation for
the largest recording dense probe biophysical model and evoked condition
OverflowError: integer 5076394960 does not fit in 'int'
"""

# import libs
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

import os
import multiprocessing
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')
import spikeinterface as si
from mpi4py import MPI
import logging
import logging.config
import yaml
import time 

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/"
os.chdir(PROJ_PATH)

# my custom software
from src.nodes import utils
from src.nodes.utils import get_config
from src.nodes.validation import snr
from src.nodes.validation import amplitude as amp

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def main(rank, n_ranks):
    """Pipeline's entry point

    Args:
        rank (_type_): rank of this computer node
        n_ranks (_type_): number of configured computed nodes in .sh file
        
    Returns:
        Write amplitude-to-noise ratio pdfs for the configured experiments
        as .npy files
    """
    
    # track time
    t_start = time.time()
    t0 = time.time()
    
    logger.info("Getting dataset configs")
    
    # neuropixels ------
    # neuropixels (Marques)
    cfg_nv, _ = get_config("vivo_marques", "c26").values() 
    PREP_PATH_nv = cfg_nv["preprocessing"]["output"]["trace_file_path"]

    # neuropixels (biophysical model)
    cfg_ns, _ = get_config("silico_neuropixels", "npx_spont").values()
    PREP_PATH_ns = cfg_ns["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]

    # neuropixels (evoked biophysical model)
    cfg_ne, _ = get_config("silico_neuropixels", "npx_evoked").values()
    PREP_PATH_ne = cfg_ne["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]

    # denser probe (horvath)  ------
    # vivo (probe 1)
    cfg_hv1, _ = get_config("vivo_horvath", "probe_1").values()
    PREP_PATH_hv1 = cfg_hv1["preprocessing"]["output"]["trace_file_path"]

    # biophy. model
    # (probe 1)
    cfg_hs1, _ = get_config("silico_horvath", "concatenated/probe_1").values()
    PREP_PATH_hs1 = cfg_hs1["preprocessing"]["output"]["trace_file_path"]

    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    
    # sampling frequency
    SFREQ_NV = 30000
    SFREQ_NS = 40000
    SFREQ_NE = 20000
    SFREQ_HV = 20000
    SFREQ_HS = 20000

    # FIGURE SETTINGS
    FIG_SIZE = (1.5, 6)
    
    # experiment colors
    COLOR_NV = np.array([153, 153, 153]) / 255 # light gray
    COLOR_NS = [0.9, 0.14, 0.15] # red
    COLOR_HV = [0.2, 0.2, 0.2] # dark gray
    COLOR_HS = np.array([26, 152, 80]) / 255 # green
    COLOR_NB = [0.22, 0.5, 0.72] # blue
    COLOR_NE = [1, 0.49, 0]  # orange

    # axes aesthetics
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 6  # 5-7 with Nature neuroscience as reference
    plt.rcParams["lines.linewidth"] = 0.5 # typically 0.5 - 1 pt
    plt.rcParams["axes.linewidth"] = 0.5 # typically 0.5 - 1 pt
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["xtick.major.width"] = 0.5 #0.8 #* 1.3
    plt.rcParams["xtick.minor.width"] = 0.5 #0.8 #* 1.3
    plt.rcParams["ytick.major.width"] = 0.5 #0.8 #* 1.3
    plt.rcParams["ytick.minor.width"] = 0.5 #0.8 #* 1.3
    plt.rcParams["xtick.major.size"] = 3.5 * 1.1
    plt.rcParams["xtick.minor.size"] = 2 * 1.1
    plt.rcParams["ytick.major.size"] = 3.5 * 1.1
    plt.rcParams["ytick.minor.size"] = 2 * 1.1
    N_MAJOR_TICKS = 4
    N_MINOR_TICKS = 12
    savefig_cfg = {"transparent":True, "dpi":300}
    legend_cfg = {"frameon": False, "handletextpad": 0.1}
    tight_layout_cfg = {"pad": 0.5}
    LG_FRAMEON = False              # no legend frame

    # cpus, gpu and current memory usage
    print("available cpus:", multiprocessing.cpu_count())
    print("available gpus:", torch.cuda.is_available())

    # 1 - Load silico and vivo traces
    # neuropixels
    RecNS = si.load_extractor(PREP_PATH_ns)
    RecNV = si.load_extractor(PREP_PATH_nv)
    RecNE = si.load_extractor(PREP_PATH_ne)
    # denser probe
    RecHV1 = si.load_extractor(PREP_PATH_hv1)
    RecHS1 = si.load_extractor(PREP_PATH_hs1)
    
    # # test 30 secs for speed
    # RecNS = RecNS.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_NS)
    # RecNV = RecNV.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_NV)
    # RecNE = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_NE)
    # # probe
    # RecHV1 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HV)
    # RecHS1 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HS)

    # 3. get traces
    # neuropixels
    traces_ns = RecNS.get_traces()
    traces_nv = RecNV.get_traces()
    traces_ne = RecNE.get_traces()
    # denser probe
    traces_hv1 = RecHV1.get_traces()
    traces_hs1 = RecHS1.get_traces()

    # 4 - get sites in L1
    # neuropixels
    lyrs = ["L1"]
    # silico
    site_ly_ns = RecNS.get_property("layers")
    sites_ns = np.where(np.isin(site_ly_ns, lyrs))[0]
    # evoked
    site_ly_ne = RecNE.get_property("layers")
    sites_ne = np.where(np.isin(site_ly_ne, lyrs))[0]  
    # vivo
    site_ly_nv = RecNV.get_property("layers")
    sites_nv = np.where(np.isin(site_ly_nv, lyrs))[0]

    # horvath
    # silico probe 1
    site_ly_hs1 = RecHS1.get_property("layers")
    sites_hs1 = np.where(np.isin(site_ly_hs1, lyrs))[0]
    # vivo probe 1
    site_ly_hv1 = RecHV1.get_property("layers")
    sites_hv1 = np.where(np.isin(site_ly_hv1, lyrs))[0]

    # track time
    t0 = time.time()
    logger.info(f"Started computing/saving amplitude-to-noise ratios on {n_ranks} nodes")
    
    # initialize the data
    max_anr, min_anr = 0, 0
    snr_ns, snr_nv, snr_ne = None, None, None
    snr_hv1 = None
    snr_hs1 = None
        
    # compute snrs and save (neuropixels probe, requires 4 computer nodes)
    logger.info(f"Started on rank {rank}")
    if rank == 0:
        snr_ns = snr.get_snrs_parallel(traces_ns[:, sites_ns]).astype(np.float32)
        max_anr = np.max(snr_ns)
        min_anr = np.min(snr_ns)
    elif rank == 1:
        snr_nv = snr.get_snrs_parallel(traces_nv[:, sites_nv]).astype(np.float32)
        max_anr = np.max(snr_nv)
        min_anr = np.min(snr_nv)
    elif rank == 2:
        snr_ne = snr.get_snrs_parallel(traces_ne[:, sites_ne]).astype(np.float32)
        max_anr = np.max(snr_ne)
        min_anr = np.min(snr_ne)
    elif rank == 3:
        snr_hv1 = snr.get_snrs_parallel(traces_hv1[:, sites_hv1]).astype(np.float32)
        max_anr = np.max(snr_hv1)
        min_anr = np.min(snr_hv1)
    elif rank == 4:
        snr_hs1 = snr.get_snrs_parallel(traces_hs1[:, sites_hs1]).astype(np.float32)
        max_anr = np.max(snr_hs1)
        min_anr = np.min(snr_hs1)
        max_anr = np.max(snr_hs1)
        min_anr = np.min(snr_hs1)
    logger.info(f"Running on rank {rank} done in {np.round(time.time()-t0,2)} secs")    

    # gather max_anr and min_anr as lists
    # node 0 data is at entry 0, 1 at 1 ...
    max_anr = comm.gather(max_anr, root=0)
    min_anr = comm.gather(min_anr, root=0)
    logger.info("Master gathered max_anr and min_anr.")

    # initialize data
    bins = None
    if rank == 0:
        # get the common bins across all experiments
        # gather the mins and maxs computed from all nodes and 
        # compute the common bins on node 0
        N_BINS = 100
        anr_max = np.max(max_anr)
        anr_min = np.min(min_anr)
        step = (anr_max - anr_min) / N_BINS
        bins = np.arange(anr_min, anr_max + step / 2, step)
        logger.info(f"Rank {rank} - bins: {bins}")
        
    # send the bins to all nodes
    # repeat to send the same bins to each node
    bins = [bins]*n_ranks
    bins = comm.scatter(bins, root=0)
    
    # track time
    t0 = time.time()
 
    # track time
    t0 = time.time()
    
    # free space
    del traces_ns
    del traces_nv
    del traces_ne
    del traces_hv1
    del traces_hs1
            
    # gather data on the master node 0
    # note: comm.gather() collects the data in a list such that the data
    # from the node 0 is at entry 0, from the node 1 at 1, node 2 at 2...
    # in the list
    snr_ns = np.array(snr_ns)
    snr_nv = comm.gather(snr_nv, root=0)
    snr_ne = comm.gather(snr_ne, root=0)
    snr_hv1 = comm.gather(snr_hv1, root=0)
    snr_hs1 = comm.gather(snr_hs1, root=0)

    logger.info(f"Done gathering stats on master node 0 in {np.round(time.time()-t0, 2)} secs")
        
    # plot and save stats on node 0
    if rank == 0:
        
        # get from node rank 1, 2 ..
        # node that produced that data 1, 2
        snr_nv = snr_nv[1]
        snr_ne = snr_ne[2]
        snr_hv1 = snr_hv1[3]
        snr_hs1 = snr_hs1[4]

        # unit-test array
        assert isinstance(snr_ns, np.ndarray)
        assert isinstance(snr_nv, np.ndarray)

        # track time
        t0 = time.time()
        
        logger.info("Started ANR plot on node 0...")
        
        # set parameters
        pm = {
            "linestyle": "-",
            "linewidth": 1,
            "marker": "None",
        }

        # plot
        FIG_SIZE = (1, 1)
        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

        # neuropixels
        ax, _ = snr.plot_layer_snr_npx(
            ax,
            "L1",
            site_ly_ns[sites_ns],
            site_ly_nv[sites_nv],
            site_ly_ne[sites_ne],
            snr_ns,
            snr_nv,
            snr_ne,
            100,
            COLOR_NS,
            COLOR_NV,
            COLOR_NE,
            "npx (Biophy.)",
            "npx (Marques-Smith)",
            "npx (Biophy. evoked)",
            pm,
        )

        # Horvath
        # - we make sure that the biophy model SNR is calculated
        # in L2/3 against the in vivo sites from the same depth and layer
        # (probe 1, L2/3)
        ax, _ = snr.plot_layer_snr_horv(
            ax,
            "L1",
            site_ly_hs1[sites_hs1],  # probe 1
            site_ly_hv1[sites_hv1],  # probe 1
            snr_hs1,  # probe 1
            snr_hv1,  # probe 1
            100,
            COLOR_HS,
            COLOR_HV,
            "custom (Biophy.)",
            "custom (Horvath)",
            pm,
        )
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([np.floor(xmin), 0, np.ceil(xmax)], [np.floor(xmin), 0, np.ceil(xmax)])
        ax.set_xlim([np.floor(xmin), np.ceil(xmax)])

        # tighten
        fig.tight_layout(**tight_layout_cfg)

        # save
        plt.savefig("anr_full_l1.svg", **savefig_cfg)

        logger.info("Saved ANR plot.")
        logger.info(f"All completed in {np.round(time.time()-t_start,2)} secs")
        
# track time
t0 = time.time()

# get api
comm = MPI.COMM_WORLD

# get the rank of this node
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# run
main(rank, n_ranks)