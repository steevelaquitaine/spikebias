"""Compute and save peak amplitude-to-noise ratio

author: steeve.laquitaine@epfl.ch

Usage:

    sbatch cluster/validation/main/snr/snr.sh
    
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
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/"
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
    t0 = time.time()
    
    logger.info("Getting dataset configs")
    
    # neuropixels ------
    # neuropixels (Marques)
    cfg_nv, _ = get_config("vivo_marques", "c26").values() 
    PREP_PATH_nv = cfg_nv["preprocessing"]["output"]["trace_file_path"]
    SNR_PATH_full_nv = cfg_nv["validation"]["full"]["trace_snr"]

    # neuropixels (biophysical model)
    cfg_ns, _ = get_config("silico_neuropixels", "concatenated").values()
    PREP_PATH_ns = cfg_ns["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]
    SNR_PATH_full_ns = cfg_ns["validation"]["full"]["trace_snr_adj10perc_less_noise_fitd_int16"]

    # neuropixels (evoked biophysical model)
    cfg_ne, _ = get_config("silico_neuropixels", "stimulus").values()
    PREP_PATH_ne = cfg_ne["preprocessing"]["output"]["full"]["trace_file_path_gain_fitd_adj10perc_less_noise_fitd_int16"]
    SNR_PATH_full_ne = cfg_ne["validation"]["full"]["trace_snr"]

    # neuropixels (synthetic, Buccino)
    cfg_nb, _ = get_config("buccino_2020", "2020").values()
    PREP_PATH_nb = cfg_nb["preprocessing"]["output"]["trace_file_path_gain_ftd"]
    SNR_PATH_full_nb = cfg_nb["validation"]["full"]["trace_snr"]

    # denser probe (horvath)  ------
    # vivo (probe 1)
    cfg_hv1, _ = get_config("vivo_horvath", "probe_1").values()
    PREP_PATH_hv1 = cfg_hv1["preprocessing"]["output"]["trace_file_path"]
    SNR_PATH_full_hv1 = cfg_hv1["validation"]["full"]["trace_snr"]
    # probe 2
    cfg_hv2, _ = get_config("vivo_horvath", "probe_2").values()
    PREP_PATH_hv2 = cfg_hv2["preprocessing"]["output"]["trace_file_path"]
    SNR_PATH_full_hv2 = cfg_hv2["validation"]["full"]["trace_snr"]
    # probe 3
    cfg_hv3, _ = get_config("vivo_horvath", "probe_3").values()
    PREP_PATH_hv3 = cfg_hv3["preprocessing"]["output"]["trace_file_path"]
    SNR_PATH_full_hv3 = cfg_hv3["validation"]["full"]["trace_snr"]

    # biophy. model
    # (probe 1)
    cfg_hs1, _ = get_config("silico_horvath", "concatenated/probe_1").values()
    PREP_PATH_hs1 = cfg_hs1["preprocessing"]["output"]["trace_file_path"]
    SNR_PATH_full_hs1 = cfg_hs1["validation"]["full"]["trace_snr"]
    # probe 2
    cfg_hs2, _ = get_config("silico_horvath", "concatenated/probe_2").values()
    PREP_PATH_hs2 = cfg_hs2["preprocessing"]["output"]["trace_file_path"]
    SNR_PATH_full_hs2 = cfg_hs2["validation"]["full"]["trace_snr"]
    # probe 3
    cfg_hs3, _ = get_config("silico_horvath", "concatenated/probe_3").values()
    PREP_PATH_hs3 = cfg_hs3["preprocessing"]["output"]["trace_file_path"]
    SNR_PATH_full_hs3 = cfg_hs3["validation"]["full"]["trace_snr"]

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
    RecNB = si.load_extractor(PREP_PATH_nb)
    RecNE = si.load_extractor(PREP_PATH_ne)
    # horvath
    # probe 1
    RecHS1 = si.load_extractor(PREP_PATH_hs1)
    RecHV1 = si.load_extractor(PREP_PATH_hv1)
    # probe 2
    RecHS2 = si.load_extractor(PREP_PATH_hs2)
    RecHV2 = si.load_extractor(PREP_PATH_hv2)
    # probe 3
    RecHS3 = si.load_extractor(PREP_PATH_hs3)
    RecHV3 = si.load_extractor(PREP_PATH_hv3)

    # 3. get traces
    # neuropixels
    traces_ns = RecNS.get_traces()
    traces_nv = RecNV.get_traces()
    traces_nb = RecNB.get_traces()
    traces_ne = RecNE.get_traces()
    # horvath
    # probe 1
    traces_hs1 = RecHS1.get_traces()
    traces_hv1 = RecHV1.get_traces()
    # probe 2
    traces_hs2 = RecHS2.get_traces()
    traces_hv2 = RecHV2.get_traces()
    # probe 3
    traces_hs3 = RecHS3.get_traces()
    traces_hv3 = RecHV3.get_traces()

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
    logger.info(f"Started to compute and save amplitude-to-noise ratios on {n_ranks} nodes")
    
    # initialize output data
    max_anr = 0
    min_anr = 0
    
    # compute snrs and save (neuropixels probe, requires 4 computer nodes)
    if rank == 0:
        logger.info(f"Started on rank 0")
        snr_ns = snr.get_snrs_parallel(traces_ns[:, sites_ns]).astype(np.float32)
        max_anr = np.max(snr_ns)
        min_anr = np.min(snr_ns)
        #utils.write_npy(snr_ns, SNR_PATH_full_ns)
        logger.info(f"Running on rank 0 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 1:
        logger.info(f"Started on rank 1")
        snr_nv = snr.get_snrs_parallel(traces_nv[:, sites_nv]).astype(np.float32)
        max_anr = np.max(snr_nv)
        min_anr = np.min(snr_nv)
        #utils.write_npy(snr_nv, SNR_PATH_full_nv)
        logger.info(f"Running on rank 1 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 2:
        logger.info(f"Started on rank 2")
        snr_nb = snr.get_snrs_parallel(traces_nb).astype(np.float32)
        max_anr = np.max(snr_nb)
        min_anr = np.min(snr_nb)
        #utils.write_npy(snr_nb, SNR_PATH_full_nb)
        logger.info(f"Running on rank 2 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 3:
        logger.info(f"Started on rank 3")
        snr_ne = snr.get_snrs_parallel(traces_ne[:, sites_ne]).astype(np.float32)
        max_anr = np.max(snr_ne)
        min_anr = np.min(snr_ne)
        #utils.write_npy(snr_ne, SNR_PATH_full_ne)
        logger.info(f"Running on rank 3 done in {np.round(time.time()-t0,2)} secs")
    # compute and save (denser probe, requires 4 computer nodes)
    elif rank == 4:
        logger.info(f"Started on rank 4")
        snr_hs1 = snr.get_snrs_parallel(traces_hs1[:, sites_hs1]).astype(np.float32)
        max_anr = np.max(snr_hs1)
        min_anr = np.min(snr_hs1)
        #utils.write_npy(snr_hs1, SNR_PATH_full_hs1)
        logger.info(f"Running on rank 4 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 5:
        logger.info(f"Started on rank 5")
        snr_hv1 = snr.get_snrs_parallel(traces_hv1[:, sites_hv1]).astype(np.float32)
        max_anr = np.max(snr_hv1)
        min_anr = np.min(snr_hv1)
        #utils.write_npy(snr_hv1, SNR_PATH_full_hv1)
        logger.info(f"Running on rank 5 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 6:
        logger.info(f"Started on rank 6")
        snr_hs2 = snr.get_snrs_parallel(traces_hs2[:, sites_hs2]).astype(np.float32)
        max_anr = np.max(snr_hs2)
        min_anr = np.min(snr_hs2)
        #utils.write_npy(snr_hs2, SNR_PATH_full_hs2)
        logger.info(f"Running on rank 6 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 7:
        logger.info(f"Started on rank 7")
        snr_hv2 = snr.get_snrs_parallel(traces_hv2[:, sites_hv2]).astype(np.float32)
        max_anr = np.max(snr_hv2)
        min_anr = np.min(snr_hv2)
        #utils.write_npy(snr_hv2, SNR_PATH_full_hv2)
        logger.info(f"Running on rank 7 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 8:
        logger.info(f"Started on rank 8")
        snr_hs3 = snr.get_snrs_parallel(traces_hs3[:, sites_hs3]).astype(np.float32)
        max_anr = np.max(snr_hs3)
        min_anr = np.min(snr_hs3)
        #utils.write_npy(snr_hs3, SNR_PATH_full_hs3)
        logger.info(f"Running on rank 8 done in {np.round(time.time()-t0,2)} secs")
    elif rank == 9:
        logger.info(f"Started on rank 9")
        snr_hv3 = snr.get_snrs_parallel(traces_hv3[:, sites_hv3]).astype(np.float32)
        max_anr = np.max(snr_hv3)
        min_anr = np.min(snr_hv3)
        #utils.write_npy(snr_hv3, SNR_PATH_full_hv3)
        logger.info(f"Running on rank 9 done in {np.round(time.time()-t0,2)} secs")

    # max_anr and min_anr are lists
    max_anr = comm.gather(max_anr, root=0)
    min_anr = comm.gather(min_anr, root=0)
    logger.info("Master gathered max_anr and min_anr.")

    # initialize output data
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
        
    # send computed bins to other nodes
    # send one copy to each node
    bins = [bins]*n_ranks
    bins = comm.scatter(bins, root=0)
    
    # track time
    t0 = time.time()
    
    # Compute summary statistics
    if rank == 0:
        logger.info(f"Started computing stats on rank 0")
        mean_ns, ci_ns, _ = amp.get_snr_pdfs(snr_ns, bins)
        logger.info(f"Done computing stats on rank 0 in {time.time()-t0} secs")
    elif rank == 1:
        logger.info(f"Started computing stats on rank 1")
        mean_nv, ci_nv, _ = amp.get_snr_pdfs(snr_nv, bins)
        logger.info(f"Done computing stats on rank 1 in {time.time()-t0} secs")
    elif rank == 2:
        logger.info(f"Started computing stats on rank 2")
        mean_nb, _, _ = amp.get_snr_pdfs(snr_nb, bins)
        logger.info(f"Done computing stats on rank 2 in {time.time()-t0} secs")
    elif rank == 3:
        logger.info(f"Started computing stats on rank 3")
        mean_ne, ci_ne, _ = amp.get_snr_pdfs(snr_ne, bins)
        logger.info(f"Done computing stats on rank 3 in {time.time()-t0} secs")
    # denser probe (horvath)
    elif rank == 4:
        logger.info(f"Started computing stats on rank 4")
        mean_hs1, _, by_site_hs1 = amp.get_snr_pdfs(snr_hs1, bins)
        logger.info(f"Done computing stats on rank 4 in {time.time()-t0} secs")
    elif rank == 5:
        logger.info(f"Started computing stats on rank 5")
        mean_hv1, _, by_site_hv1 = amp.get_snr_pdfs(snr_hv1, bins)
        logger.info(f"Done computing stats on rank 5 in {time.time()-t0} secs")
    elif rank == 6:
        logger.info(f"Started computing stats on rank 6")
        mean_hs2, _, by_site_hs2 = amp.get_snr_pdfs(snr_hs2, bins)
        logger.info(f"Ended computing stats on rank 6")
        logger.info(f"Done computing stats on rank 6 in {time.time()-t0} secs")
    elif rank == 7:
        logger.info(f"Started computing stats on rank 7")
        mean_hv2, _, by_site_hv2 = amp.get_snr_pdfs(snr_hv2, bins)
        logger.info(f"Done computing stats on rank 7 in {time.time()-t0} secs")
    elif rank == 8:
        logger.info(f"Started computing stats on rank 8")
        mean_hs3, _, by_site_hs3 = amp.get_snr_pdfs(snr_hs3, bins)
        logger.info(f"Done computing stats on rank 8 in {time.time()-t0} secs")
    elif rank == 9:
        logger.info(f"Started computing stats on rank 9")
        mean_hv3, _, by_site_hv3 = amp.get_snr_pdfs(snr_hv3, bins)
        logger.info(f"Done computing stats on rank 9 in {time.time()-t0} secs")
        
    # plot and save stats on node 0
    if rank == 0:
        
        # track time
        t0 = time.time()
        logger.info("Gathering all stats on rank 0")

        # gather summary data on node 0
        mean_ns = comm.gather(mean_ns, root=0)
        mean_nv = comm.gather(mean_nv, root=0)
        mean_nb = comm.gather(mean_nb, root=0)
        mean_ne = comm.gather(mean_ne, root=0)
        mean_hs1 = comm.gather(mean_hs1, root=0)
        mean_hv1 = comm.gather(mean_hv1, root=0)
        mean_hs2 = comm.gather(mean_hs2, root=0)
        mean_hv2 = comm.gather(mean_hv2, root=0)
        mean_hs3 = comm.gather(mean_hs3, root=0)
        mean_hv3 = comm.gather(mean_hv3, root=0)
        
        ci_ns = comm.gather(ci_ns, root=0)
        ci_nv = comm.gather(ci_nv, root=0)
        ci_ne = comm.gather(ci_ne, root=0)
            
        by_site_hv1 = comm.gather(by_site_hv1, root=0)
        by_site_hv2 = comm.gather(by_site_hv2, root=0)
        by_site_hv3 = comm.gather(by_site_hv3, root=0)
        by_site_hs1 = comm.gather(by_site_hs1, root=0)
        by_site_hs2 = comm.gather(by_site_hs2, root=0)
        by_site_hs3 = comm.gather(by_site_hs3, root=0)
        
        logger.info(f"Gathered all stats on rank 0 in {np.round(time.time()-t0, 2)} secs")

        # Compute mean and CI over all sites pooling L1, L2/3 (probe 1), L4,5 (probe 2), L6 (probe 3) horvath
        # pool
        pooled_hv = (
            by_site_hv1["pdf_by_site"] + by_site_hv2["pdf_by_site"] + by_site_hv3["pdf_by_site"]
        )
        pooled_hs = (
            by_site_hs1["pdf_by_site"] + by_site_hs2["pdf_by_site"] + by_site_hs3["pdf_by_site"]
        )
        # get mean and ci
        mean_hv, ci_hv = snr.get_pdf_mean_ci(pooled_hv)
        mean_hs, ci_hs = snr.get_pdf_mean_ci(pooled_hs)

        # unit-test probabilities
        # neuropixels
        assert 1 - sum(mean_nv) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_ns) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_nb) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_ne) < 1e-15, "should sum to 1"

        # custom (horvath)
        assert 1 - sum(mean_hv1) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_hv2) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_hv3) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_hs1) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_hs2) < 1e-15, "should sum to 1"
        assert 1 - sum(mean_hs3) < 1e-15, "should sum to 1"

        logger.info("Started ANR plot...")
        
        # set parameters
        pm = {
            "linestyle": "-",
            "linewidth": 1,
            "marker": "None",
        }

        # plot
        FIG_SIZE = (1.5, 6)
        fig, ax = plt.subplots(6, 1, figsize=FIG_SIZE)

        # all sites ********************************

        # neuropixels
        ax[0] = amp.plot_snr_pdf_all(
            ax[0],
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
        ax[0] = amp.plot_snr_pdf_all(
            ax[0],
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
        xmin, xmax = ax[0].get_xlim()
        ax[0].set_xticks([np.floor(xmin), 0, np.ceil(xmax)], [np.floor(xmin), 0, np.ceil(xmax)])
        ax[0].set_xlim([np.floor(xmin), np.ceil(xmax)])
        
        # L1 ********************************

        # neuropixels
        ax[1], _ = snr.plot_layer_snr_npx(
            ax[1],
            "L1",
            site_ly_ns,
            site_ly_nv,
            site_ly_ne,
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
        # horvath
        # - we make sure that the biophy model SNR is calculated
        # in L2/3 against the in vivo sites from the same depth and layer
        # (probe 1, L2/3)
        ax[1], _ = snr.plot_layer_snr_horv(
            ax[1],
            "L1",
            site_ly_hs1,  # probe 1
            site_ly_hv1,  # probe 1
            snr_hs1,  # probe 1
            snr_hv1,  # probe 1
            100,
            COLOR_HS,
            COLOR_HV,
            "custom (Biophy.)",
            "custom (Horvath)",
            pm,
        )

        xmin, xmax = ax[1].get_xlim()
        ax[1].set_xticks([np.floor(xmin), 0, np.ceil(xmax)], [np.floor(xmin), 0, np.ceil(xmax)])
        ax[1].set_xlim([np.floor(xmin), np.ceil(xmax)])


        # L2/3 ********************************

        # neuropixels
        ax[2], _ = snr.plot_layer_snr_npx(
            ax[2],
            "L2_3",
            site_ly_ns,
            site_ly_nv,
            site_ly_ne,
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
        # custom (probe 1, layer 2/3)
        ax[2], _ = snr.plot_layer_snr_horv(
            ax[2],
            "L2_3",
            site_ly_hs1,  # probe 1
            site_ly_hv1,  # probe 1
            snr_hs1,  # probe 1
            snr_hv1,  # probe 1
            100,
            COLOR_HS,
            COLOR_HV,
            "custom (Biophy.)",
            "custom (Horvath)",
            pm,
        )
        xmin, xmax = ax[2].get_xlim()
        ax[2].set_xticks([np.floor(xmin), 0, np.ceil(xmax)], [np.floor(xmin), 0, np.ceil(xmax)])
        ax[2].set_xlim([np.floor(xmin), np.ceil(xmax)])
        

        # L4 ********************************

        # neuropixels
        ax[3], _ = snr.plot_layer_snr_npx(
            ax[3],
            "L4",
            site_ly_ns,
            site_ly_nv,
            site_ly_ne,
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
        # custom (probe 2)
        # - we make sure that the biophy model SNR is calculated
        # in L4 against the in vivo sites from the same depth and layer
        # (probe 2, L4)
        ax[3], _ = snr.plot_layer_snr_horv(
            ax[3],
            "L4",
            site_ly_hs2,  # probe 2
            site_ly_hv2,  # probe 2
            snr_hs2,  # probe 2
            snr_hv2,  # probe 2
            100,
            COLOR_HS,
            COLOR_HV,
            "custom (Biophy.)",
            "custom (Horvath)",
            pm,
        )
        
        xmin, xmax = ax[3].get_xlim()
        ax[3].set_xticks([np.floor(xmin), 0, np.ceil(xmax)], [np.floor(xmin), 0, np.ceil(xmax)])
        ax[3].set_xlim([np.floor(xmin), np.ceil(xmax)])


        # (3m) L5 ***************************

        # neuropixels
        ax[4], _ = snr.plot_snr_for_layer_5_npx(
            ax[4],
            site_ly_ns,
            site_ly_nv,
            site_ly_ne,
            snr_ns,
            snr_nv,
            snr_ne,
            snr_nb,
            100,
            COLOR_NS,
            COLOR_NV,
            COLOR_NE,
            COLOR_NB,
            "npx (Biophy.)",
            "npx (Marques)",
            "npx (Biophy. evoked)",
            "npx (Synth.)",
            pm,
        )
        # denser probe (probe 2)
        # - we make sure that the biophy model SNR is calculated
        # in L5 against the in vivo sites from the same depth and layer
        # (probe 2, L5)
        ax[4], _ = snr.plot_snr_for_layer_5_horv(
            ax[4],
            site_ly_hs2,  # probe 2
            site_ly_hv2,  # probe 2
            snr_hs2,  # probe 2
            snr_hv2,  # probe 2
            100,
            COLOR_HS,
            COLOR_HV,
            "custom (Biophy.)",
            "custom (Horvath)",
            pm,
        )

        xmin, xmax = ax[4].get_xlim()
        ax[4].set_xticks([np.floor(xmin), 0, np.ceil(xmax)], [np.floor(xmin), 0, np.ceil(xmax)])
        ax[4].set_xlim([np.floor(xmin), np.ceil(xmax)])

        # L6 ********************************

        # neuropixels
        ax[5], _ = snr.plot_layer_snr_npx(
            ax[5],
            "L6",
            site_ly_ns,
            site_ly_nv,
            site_ly_ne,
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
        # denser (probe 3)
        # - we make sure that the biophy model SNR is calculated
        # in L5 against the in vivo sites from the same depth and layer
        # (probe 3, L6)
        ax[5], _ = snr.plot_layer_snr_horv(
            ax[5],
            "L6",
            site_ly_hs3,  # probe 3
            site_ly_hv3,  # probe 3
            snr_hs3,  # probe 3
            snr_hv3,  # probe 3
            100,
            COLOR_HS,
            COLOR_HV,
            "custom (Biophy.)",
            "custom (Horvath)",
            pm,
        )
        xmin, xmax = ax[5].get_xlim()
        ax[5].set_xticks([np.floor(xmin), 0, np.ceil(xmax)], [np.floor(xmin), 0, np.ceil(xmax)])
        ax[5].set_xlim([np.floor(xmin), np.ceil(xmax)])    

        # tighten
        fig.tight_layout(**tight_layout_cfg)

        # save
        plt.savefig("figures/6_supp/fig2/fig2q_anr_full.svg", **savefig_cfg)

        logger.info("Saved ANR plot.")
        
# track time
t0 = time.time()

# get api
comm = MPI.COMM_WORLD

# get the rank of this node
rank = comm.Get_rank()
n_ranks = comm.Get_size()

# run
main(rank, n_ranks)
logger.info(f"All done in {np.round(time.time()-t0,2)} secs")