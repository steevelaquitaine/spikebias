"""Compute and save peak amplitude-to-noise ratio

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/validation/anr/anr_all_sites.sh
    
Duration: 12 minutes
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
    cfg_ns, _ = get_config("silico_neuropixels", "concatenated").values()
    PREP_PATH_ns = cfg_ns["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]

    # neuropixels (evoked biophysical model)
    cfg_ne, _ = get_config("silico_neuropixels", "stimulus").values()
    PREP_PATH_ne = cfg_ne["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]

    # denser probe (horvath)  ------
    # vivo (probe 1)
    cfg_hv1, _ = get_config("vivo_horvath", "probe_1").values()
    PREP_PATH_hv1 = cfg_hv1["preprocessing"]["output"]["trace_file_path"]
    # probe 2
    cfg_hv2, _ = get_config("vivo_horvath", "probe_2").values()
    PREP_PATH_hv2 = cfg_hv2["preprocessing"]["output"]["trace_file_path"]
    # probe 3
    cfg_hv3, _ = get_config("vivo_horvath", "probe_3").values()
    PREP_PATH_hv3 = cfg_hv3["preprocessing"]["output"]["trace_file_path"]

    # biophy. model
    # (probe 1)
    cfg_hs1, _ = get_config("silico_horvath", "concatenated/probe_1").values()
    PREP_PATH_hs1 = cfg_hs1["preprocessing"]["output"]["trace_file_path"]
    # probe 2
    cfg_hs2, _ = get_config("silico_horvath", "concatenated/probe_2").values()
    PREP_PATH_hs2 = cfg_hs2["preprocessing"]["output"]["trace_file_path"]
    # probe 3
    cfg_hs3, _ = get_config("silico_horvath", "concatenated/probe_3").values()
    PREP_PATH_hs3 = cfg_hs3["preprocessing"]["output"]["trace_file_path"]
    
    logger.info(f"Done in {np.round(time.time()-t0,2)} secs")
    
    # sampling frequency
    SFREQ_NV = 30000
    SFREQ_NS = 40000
    SFREQ_NE = 20000
    SFREQ_HV = 20000
    SFREQ_HS = 20000

    # FIGURE SETTINGS
    FIG_SIZE = (1.3, 1.3)
    
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
    
    # probe
    RecHV1 = si.load_extractor(PREP_PATH_hv1)
    RecHV2 = si.load_extractor(PREP_PATH_hv2)
    RecHV3 = si.load_extractor(PREP_PATH_hv3)
    RecHS1 = si.load_extractor(PREP_PATH_hs1)
    RecHS2 = si.load_extractor(PREP_PATH_hs2)
    RecHS3 = si.load_extractor(PREP_PATH_hs3)    
    
    # # test 30 secs for speed
    # RecNS = RecNS.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_NS)
    # RecNV = RecNV.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_NV)
    # RecNE = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_NE)
    
    # # probe
    # RecHV1 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HV)
    # RecHV2 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HV)
    # RecHV3 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HV)
    # RecHS1 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HS)
    # RecHS2 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HS)
    # RecHS3 = RecNE.frame_slice(start_frame=0, end_frame=1 * 30 * SFREQ_HS)

    # 3. get traces
    # neuropixels
    traces_ns = RecNS.get_traces()
    traces_nv = RecNV.get_traces()
    traces_ne = RecNE.get_traces()
    
    # denser probe
    traces_hv1 = RecHV1.get_traces()
    traces_hv2 = RecHV2.get_traces()
    traces_hv3 = RecHV3.get_traces()
    traces_hs1 = RecHS1.get_traces()
    traces_hs2 = RecHS2.get_traces()
    traces_hs3 = RecHS3.get_traces()

    # 4 - get good sites (in cortex)
    # neuropixels
    lyrs = ["L1", "L2_3", "L4", "L5", "L6"]
    # silico
    site_ly_ns = RecNS.get_property("layers")
    site_ly_ns[site_ly_ns == "L2"] = "L2_3"
    site_ly_ns[site_ly_ns == "L3"] = "L2_3"
    sites_ns = np.where(np.isin(site_ly_ns, lyrs))[0]
    # evoked
    site_ly_ne = RecNE.get_property("layers")
    site_ly_ne[site_ly_ne == "L2"] = "L2_3"
    site_ly_ne[site_ly_ne == "L3"] = "L2_3"
    sites_ne = np.where(np.isin(site_ly_ne, lyrs))[0]  
    # vivo
    site_ly_nv = RecNV.get_property("layers")
    sites_nv = np.where(np.isin(site_ly_nv, lyrs))[0]

    # horvath
    # silico
    # probe 1 (select L1, L2/3)
    site_ly_hs1 = RecHS1.get_property("layers")
    site_ly_hs1 = np.array(["L2_3" if x == "L2" or x == "L3" else x for x in site_ly_hs1])
    sites_hs1 = np.where(np.isin(site_ly_hs1, ["L1", "L2_3"]))[0]
    # probe 2 (select L4 and L5)
    site_ly_hs2 = RecHS2.get_property("layers")
    sites_hs2 = np.where(np.isin(site_ly_hs2, ["L4", "L5"]))[0]
    # probe 3 (select L6)
    site_ly_hs3 = RecHS3.get_property("layers")
    site_ly_hs3 = np.array(["L2_3" if x == "L2" or x == "L3" else x for x in site_ly_hs3])
    sites_hs3 = np.where(np.isin(site_ly_hs3, ["L6"]))[0]

    # vivo
    # probe 1 (select L1, L2/3)
    site_ly_hv1 = RecHV1.get_property("layers")
    sites_hv1 = np.where(np.isin(site_ly_hv1, ["L1", "L2_3"]))[0]
    # probe 2 (select L4, L5)
    site_ly_hv2 = RecHV2.get_property("layers")
    sites_hv2 = np.where(np.isin(site_ly_hv2, ["L4", "L5"]))[0]
    # probe 3 (select L6)
    site_ly_hv3 = RecHV3.get_property("layers")
    sites_hv3 = np.where(np.isin(site_ly_hv3, ["L6"]))[0]

    # track time
    t0 = time.time()
    logger.info(f"Started computing/saving amplitude-to-noise ratios on {n_ranks} nodes")
    
    # initialize the data
    max_anr, min_anr = 0, 0
    snr_ns, snr_nv, snr_ne = None, None, None
    snr_hv1, snr_hv2, snr_hv3 = None, None, None
    snr_hs1, snr_hs2, snr_hs3 = None, None, None
        
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
        snr_hv2 = snr.get_snrs_parallel(traces_hv2[:, sites_hv2]).astype(np.float32)
        max_anr = np.max(snr_hv2)
        min_anr = np.min(snr_hv2)
    elif rank == 5:
        snr_hv3 = snr.get_snrs_parallel(traces_hv3[:, sites_hv3]).astype(np.float32)
        max_anr = np.max(snr_hv3)
        min_anr = np.min(snr_hv3)
    elif rank == 6:
        snr_hs1 = snr.get_snrs_parallel(traces_hs1[:, sites_hs1]).astype(np.float32)
        max_anr = np.max(snr_hs1)
        min_anr = np.min(snr_hs1)
    elif rank == 7:
        # 6 mins
        snr_hs2 = snr.get_snrs_parallel(traces_hs2[:, sites_hs2]).astype(np.float32)
        max_anr = np.max(snr_hs2)
        min_anr = np.min(snr_hs2)
    elif rank == 8:
        snr_hs3 = snr.get_snrs_parallel(traces_hs3[:, sites_hs3]).astype(np.float32)
        max_anr = np.max(snr_hs3)
        min_anr = np.min(snr_hs3)
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
 
    # initialize output data
    mean_ns = None
    mean_nv = None
    mean_ne = None
    mean_hv1 = None
    mean_hv2 = None
    mean_hv3 = None
    mean_hs1 = None
    mean_hs2 = None
    mean_hs3 = None
    ci_ns = None
    ci_nv = None
    ci_ne = None
    ci_hv1 = None
    ci_hv2 = None
    ci_hv3 = None
    ci_hs1 = None
    ci_hs2 = None
    ci_hs3 = None
    by_site_hv1 = None
    by_site_hv2 = None
    by_site_hv3 = None
    by_site_hs1 = None
    by_site_hs2 = None
    by_site_hs3 = None

    # Compute summary statistics
    logger.info(f"Started computing stats on rank {rank}")
    if rank == 0:
        mean_ns, ci_ns, _ = amp.get_snr_pdfs(snr_ns, bins)
    elif rank == 1:
        mean_nv, ci_nv, _ = amp.get_snr_pdfs(snr_nv, bins)
    elif rank == 2:
        mean_ne, ci_ne, _ = amp.get_snr_pdfs(snr_ne, bins)
    elif rank == 3:
        mean_hv1, ci_hv1, by_site_hv1 = amp.get_snr_pdfs(snr_hv1, bins)
    elif rank == 4:
        mean_hv2, ci_hv2, by_site_hv2 = amp.get_snr_pdfs(snr_hv2, bins)
    elif rank == 5:
        mean_hv3, ci_hv3, by_site_hv3 = amp.get_snr_pdfs(snr_hv3, bins)
    elif rank == 6:
        mean_hs1, ci_hs1, by_site_hs1 = amp.get_snr_pdfs(snr_hs1, bins)
    elif rank == 7:
        mean_hs2, ci_hs2, by_site_hs2 = amp.get_snr_pdfs(snr_hs2, bins)
    elif rank == 8:
        mean_hs3, ci_hs3, by_site_hs3 = amp.get_snr_pdfs(snr_hs3, bins)
    logger.info(f"Done computing stats on rank {rank}")
    # track time
    t0 = time.time()
    
    # gather data on the master node 0
    # note: comm.gather() collects the data in a list such that the data
    # from the node 0 is at entry 0, from the node 1 at 1, node 2 at 2...
    # in the list
    mean_ns = np.array(mean_ns)
    ci_ns = np.array(ci_ns)
    
    mean_nv = comm.gather(mean_nv, root=0)
    ci_nv = comm.gather(ci_nv, root=0)

    mean_ne = comm.gather(mean_ne, root=0)
    ci_ne = comm.gather(ci_ne, root=0)
    
    mean_hv1 = comm.gather(mean_hv1, root=0)
    ci_hv1 = comm.gather(ci_hv1, root=0)
    by_site_hv1 = comm.gather(by_site_hv1, root=0)

    mean_hv2 = comm.gather(mean_hv2, root=0)
    ci_hv2 = comm.gather(ci_hv2, root=0)
    by_site_hv2 = comm.gather(by_site_hv2, root=0)

    mean_hv3 = comm.gather(mean_hv3, root=0)
    ci_hv3 = comm.gather(ci_hv3, root=0)
    by_site_hv3 = comm.gather(by_site_hv3, root=0)
                
    mean_hs1 = comm.gather(mean_hs1, root=0)
    ci_hs1 = comm.gather(ci_hs1, root=0)
    by_site_hs1 = comm.gather(by_site_hs1, root=0)

    mean_hs2 = comm.gather(mean_hs2, root=0)
    ci_hs2 = comm.gather(ci_hs2, root=0)
    by_site_hs2 = comm.gather(by_site_hs2, root=0)

    mean_hs3 = comm.gather(mean_hs3, root=0)
    ci_hs3 = comm.gather(ci_hs3, root=0)
    by_site_hs3 = comm.gather(by_site_hs3, root=0)          
    
    logger.info(f"Done gathering stats on master node 0 in {np.round(time.time()-t0, 2)} secs")
        
    # plot and save stats on node 0
    if rank == 0:
        
        # get from node rank 1, 2 ..
        # node that produced that data 1, 2
        mean_nv = mean_nv[1]
        ci_nv = ci_nv[1]
                
        mean_ne = mean_ne[2]
        ci_ne = ci_ne[2]
        
        mean_hv1 = mean_hv1[3]
        ci_hv1 = ci_hv1[3]
        by_site_hv1 = by_site_hv1[3]

        mean_hv2 = mean_hv2[4]
        ci_hv2 = ci_hv2[4]
        by_site_hv2 = by_site_hv2[4]

        mean_hv3 = mean_hv3[5]
        ci_hv3 = ci_hv3[5]
        by_site_hv3 = by_site_hv3[5]

        mean_hs1 = mean_hs1[6]
        ci_hs1 = ci_hs1[6]
        by_site_hs1 = by_site_hs1[6]

        mean_hs2 = mean_hs2[7]
        ci_hs2 = ci_hs2[7]
        by_site_hs2 = by_site_hs2[7]

        mean_hs3 = mean_hs3[8]
        ci_hs3 = ci_hs3[8]
        by_site_hs3 = by_site_hs3[8]

        # unit-test array
        assert isinstance(mean_ns, np.ndarray)
        assert isinstance(ci_ns, np.ndarray)
        assert isinstance(mean_nv, np.ndarray)
        assert isinstance(ci_nv, np.ndarray)

        # # unit-test probabilities
        logger.info(f"sum(mean_nv): {sum(mean_nv)} should be close to 1")
        # assert 1 - sum(mean_nv) < 1e-15, "should sum to 1"
        # assert 1 - sum(mean_ns) < 1e-15, "should sum to 1"
        # assert 1 - sum(mean_ne) < 1e-15, "should sum to 1"

        # track time
        t0 = time.time()
        
        # Compute mean and CI over all sites pooling L1, L2/3 (probe 1), L4,5 (probe 2), L6 (probe 3) horvath
        # pool
        pooled_hv = (
            by_site_hv1["pdf_by_site"] + by_site_hv2["pdf_by_site"] + by_site_hv3["pdf_by_site"]
        )
        pooled_hs = (
            by_site_hs1["pdf_by_site"] + by_site_hs2["pdf_by_site"] + by_site_hs3["pdf_by_site"]
        )
        # get median and 95% ci
        mean_hv, ci_hv = snr.get_pdf_median_ci(pooled_hv)
        mean_hs, ci_hs = snr.get_pdf_median_ci(pooled_hs)
        
        logger.info(f"Done pooling dense probe depths in {np.round(time.time()-t0, 2)} secs")
        
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
        fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

        # all sites ********************************

        # neuropixels
        ax = amp.plot_snr_pdf_all(
            ax,
            mean_nv,
            mean_ns,
            mean_ne,
            ci_nv,
            ci_ns,
            ci_ne,
            bins,
            COLOR_NV,
            COLOR_NS,
            COLOR_NE,
            pm,
        )
        
        # Horvath
        ax = amp.plot_snr_pdf_all(
            ax,
            mean_hv,
            mean_hs,
            [0],
            ci_hv,
            ci_hs,
            [0],
            bins,
            COLOR_HV,
            COLOR_HS,
            [0],
            pm,
        )
        xmin, xmax = ax.get_xlim()
        ax.set_xticks([np.floor(xmin).astype(int), 0, np.ceil(xmax).astype(int)], [np.floor(xmin).astype(int), 0, np.ceil(xmax).astype(int)])
        ax.set_xlim([np.floor(xmin), np.ceil(xmax)])
        
        # tighten
        fig.tight_layout(**tight_layout_cfg)

        # save
        plt.savefig("figures/6_supp/fig2/fig2f_anr_full_all_sites.svg", **savefig_cfg)
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