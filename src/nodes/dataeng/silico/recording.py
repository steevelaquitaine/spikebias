"""nodes that process raw traces as SpikeInterface Recording Extractor objects

Usage:

    # activate your spack environment
    module load unstable hpe-mpi/2.25.hmpt matlab
    module load spack
    cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
    . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
    spack env activate spack_env -p
    spack load python@3.9.7

    # run
    
Returns:
    _type_: _description_
"""
import os
import logging
import logging.config
import shutil
from sys import argv
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.full as si
import yaml
import copy
import time
from src.nodes.load import load_campaign_params
import torch
import json

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def _load_stored_fit_results(data_conf: dict):
    """load results from fitting silico trace amplitude and noise to 
    in vivo traces per layer

    Args:
        data_conf (dict): _description_

    Returns:
        list: list of dictionaries containing fit results per layer
    """
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]

    # if already fitted data exist
    # return concatenated noise per layer
    noises = ()
    if os.path.isfile(NOISE_PATH + "L1.npy"):
        noises += (np.load(NOISE_PATH + "L1.npy", allow_pickle=True).item(),)
    if os.path.isfile(NOISE_PATH + "L2_3.npy"):
        noises += (np.load(NOISE_PATH + "L2_3.npy", allow_pickle=True).item(),)
    if os.path.isfile(NOISE_PATH + "L4.npy"):
        noises += (np.load(NOISE_PATH + "L4.npy", allow_pickle=True).item(),)
    if os.path.isfile(NOISE_PATH + "L5.npy"):
        noises += (np.load(NOISE_PATH + "L5.npy", allow_pickle=True).item(),)
    if os.path.isfile(NOISE_PATH + "L6.npy"):
        noises += (np.load(NOISE_PATH + "L6.npy", allow_pickle=True).item(),)
    return noises


def _load_stored_fit_results_nwb(data_conf: dict):
    """load results from fitting silico trace amplitude and noise to 
    in vivo traces per layer

    Args:
        data_conf (dict): _description_

    Returns:
        list: list of dictionaries containing fit results per layer
    """
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]

    # return concatenated noise per layer
    noises = ()
    layers = ["L1", "L2_3", "L4", "L5", "L6"]
    for layer in layers:
        if os.path.isfile(NOISE_PATH + f"{layer}_tuned.json"):
            with open(NOISE_PATH + f"{layer}_tuned.json", 'r') as file:
                noises += (json.load(file),)
    return noises


def _load_stored_fit_results2(noise_path):
    """load results from fitting silico trace amplitude and noise to 
    in vivo traces per layer
    noise is reduced by 20%

    Args:
        data_conf (dict): _description_

    Returns:
        list: list of dictionaries containing fit results per layer
    """
    noises = ()
    if os.path.isfile(noise_path + "L1.npy"):
        noises += (np.load(noise_path + "L1.npy", allow_pickle=True).item(),)
    if os.path.isfile(noise_path + "L2_3.npy"):
        noises += (np.load(noise_path + "L2_3.npy", allow_pickle=True).item(),)
    if os.path.isfile(noise_path + "L4.npy"):
        noises += (np.load(noise_path + "L4.npy", allow_pickle=True).item(),)
    if os.path.isfile(noise_path + "L5.npy"):
        noises += (np.load(noise_path + "L5.npy", allow_pickle=True).item(),)
    if os.path.isfile(noise_path + "L6.npy"):
        noises += (np.load(noise_path + "L6.npy", allow_pickle=True).item(),)
    return noises


def create_noise_matrix(noises, n_sites, n_samples):

    # set reproducibility
    torch.manual_seed(noises[0]["seed"])
    np.random.seed(noises[0]["seed"])

    # assign noise rms to each site
    # - zeros will be added to sites outside cortex (mean=0, std=0)
    # - noise_rms is a column-vector of n sites
    noise_rms = torch.tensor(0).repeat(n_sites, 1)
    for ix, _ in enumerate(noises):
        noise_rms[noises[ix]["layer_sites_ix"]] = noises[ix]["missing_noise_rms"]

    # unit-test
    assert all(np.isnan(noise_rms)) == False, "there should be no nan values"
    return torch.randn(n_sites, n_samples) * noise_rms


def _scale_and_add_noise(traces: np.array, data_conf: dict):
    """load gain and missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """

    # get stored results for the best fit scaling factor and missing noise
    fit_out = _load_stored_fit_results(data_conf)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_adj_and_add_noise(traces: np.array, data_conf: dict):
    """load gain and missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """

    # get stored results for the best fit scaling factor and missing noise
    fit_out = _load_stored_fit_results(data_conf)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"] * np.sqrt(2) # slightly increase gain
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_adj_by_and_add_noise(traces: np.array, data_conf: dict, gain_prms:dict):
    """load gain and missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # get fitted results
    fit_out = _load_stored_fit_results(data_conf)

    # get the tuned gain
    gain = fit_out[0]["gain"] * gain_prms["gain_adjust"]
    
    # create missing noise
    missing_noise = create_noise_matrix(fit_out, traces.shape[1], traces.shape[0])
    
    # add noise and cast as array
    traces = torch.from_numpy(traces)
    fitted_traces = (traces.T * gain) + missing_noise
    fitted_traces = fitted_traces.cpu().detach().numpy()
    return fitted_traces.T


def _scale_adj_by_and_add_noise_nwb(traces: np.array, data_conf: dict, gain_prms:dict):
    """load gain and missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # get fitted results
    fit_out = _load_stored_fit_results_nwb(data_conf)    

    # get the tuned gain
    gain = fit_out[0]["gain"] * gain_prms["gain_adjust"]
    
    # create missing noise
    logger.info(f"Creating noise matrix ...")
    missing_noise = create_noise_matrix(fit_out, traces.shape[1], traces.shape[0])
    logger.info(f"Done creating noise matrix.")
    
    # add noise and cast as array
    traces = torch.from_numpy(traces)
    fitted_traces = (traces.T * gain) + missing_noise
    fitted_traces = fitted_traces.cpu().detach().numpy()
    return fitted_traces.T


def _scale_and_add_noise_20_perc_lower(traces: np.array, data_conf: dict):
    """load gain and missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """

    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_20_perc_lower_path"]    

    # get stored results for the best fit scaling factor and missing noise
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _save_50_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_50_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_50_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 50 %
    l1_out["missing_noise_rms"] = 0.5 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.5 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.5 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.5 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.5 * l6_out["missing_noise_rms"]

    # save 
    np.save(NOISE_50_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_50_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_50_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_50_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_50_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 50 percent reduced noise files.")


def _save_75_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_75_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_75_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 75 %
    l1_out["missing_noise_rms"] = 0.25 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.25 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.25 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.25 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.25 * l6_out["missing_noise_rms"]

    # save
    np.save(NOISE_75_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 75 percent reduced noise files.")


def _save_80_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_80_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_80_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 75 %
    l1_out["missing_noise_rms"] = 0.2 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.2 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.2 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.2 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.2 * l6_out["missing_noise_rms"]

    # save
    np.save(NOISE_80_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_80_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_80_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_80_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_80_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 80 percent reduced noise files.")


def _save_90_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_90_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_90_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 75 %
    l1_out["missing_noise_rms"] = 0.1 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.1 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.1 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.1 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.1 * l6_out["missing_noise_rms"]

    # save
    np.save(NOISE_90_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_90_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_90_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_90_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_90_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 90 percent reduced noise files.")
    
    
def _save_95_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_95_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_95_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 75 %
    l1_out["missing_noise_rms"] = 0.05 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.05 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.05 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.05 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.05 * l6_out["missing_noise_rms"]

    # save
    np.save(NOISE_95_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_95_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_95_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_95_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_95_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 95 percent reduced noise files.")
    
    
def _save_99_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_99_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_99_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 75 %
    l1_out["missing_noise_rms"] = 0.01 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.01 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.01 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.01 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.01 * l6_out["missing_noise_rms"]

    # save
    np.save(NOISE_99_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_99_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_99_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_99_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_99_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 99 percent reduced noise files.")    
    
    
def _save_40_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_75_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_40_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 75 %
    l1_out["missing_noise_rms"] = 0.60 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.60 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.60 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.60 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.60 * l6_out["missing_noise_rms"]

    # save
    np.save(NOISE_75_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_75_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 40 percent lower noise files.")


def _save_60_perc_lower_noise(data_conf:dict):
    
    # get fitted noise path
    FIT_NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_path"]
    NOISE_60_PERC_SAVE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_60_perc_lower_path"]
    
    # load fitted noises
    l1_out = np.load(FIT_NOISE_PATH + "L1.npy", allow_pickle=True).item()
    l23_out = np.load(FIT_NOISE_PATH + "L2_3.npy", allow_pickle=True).item()
    l4_out = np.load(FIT_NOISE_PATH + "L4.npy", allow_pickle=True).item()
    l5_out = np.load(FIT_NOISE_PATH + "L5.npy", allow_pickle=True).item()
    l6_out = np.load(FIT_NOISE_PATH + "L6.npy", allow_pickle=True).item()

    # reduce noise by 75 %
    l1_out["missing_noise_rms"] = 0.40 * l1_out["missing_noise_rms"]
    l23_out["missing_noise_rms"] = 0.40 * l23_out["missing_noise_rms"]
    l4_out["missing_noise_rms"] = 0.40 * l4_out["missing_noise_rms"]
    l5_out["missing_noise_rms"] = 0.40 * l5_out["missing_noise_rms"]
    l6_out["missing_noise_rms"] = 0.40 * l6_out["missing_noise_rms"]

    # save
    np.save(NOISE_60_PERC_SAVE_PATH + "L1.npy", l1_out)
    np.save(NOISE_60_PERC_SAVE_PATH + "L2_3.npy", l23_out)
    np.save(NOISE_60_PERC_SAVE_PATH + "L4.npy", l4_out)
    np.save(NOISE_60_PERC_SAVE_PATH + "L5.npy", l5_out)
    np.save(NOISE_60_PERC_SAVE_PATH + "L6.npy", l6_out)
    logger.info("Saved 60 percent lower noise files.")


def _scale_and_add_noise_50_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 50 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 50% fitted noise files 
    _save_50_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_50_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_and_add_noise_75_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 75 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 50% fitted noise files 
    _save_75_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_75_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_and_add_noise_80_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 75 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 50% fitted noise files 
    _save_80_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_80_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_and_add_noise_90_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 90 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 50% fitted noise files 
    _save_90_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_90_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_and_add_noise_95_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 95 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 50% fitted noise files 
    _save_95_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_95_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_and_add_noise_99_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 99 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 99% fitted noise files 
    _save_99_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_99_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_and_add_noise_40_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 40 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 60% of (40% lower) fitted noise files 
    _save_40_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_40_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale_and_add_noise_60_perc_lower(traces: np.array, data_conf: dict):
    """load gain and add 60 percent of missing noise best fitted
    to the in vivo traces per layer and transform silico traces

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    # save 60% of (40% lower) fitted noise files 
    _save_60_perc_lower_noise(data_conf)
    
    # get config
    NOISE_PATH = data_conf["preprocessing"]["fitting"]["missing_noise_60_perc_lower_path"]

    # load saved files in a list of dicts
    fit_out = _load_stored_fit_results2(NOISE_PATH)
    
    # set seed for reproducibility
    np.random.seed(fit_out[0]["seed"])

    # make writable (40s/h recording)
    fitted_traces = copy.copy(traces).T
    nsites = traces.shape[1]
    ntimepoints = traces.shape[0]

    # - scale trace and add missing noise to each site
    for ix, _ in enumerate(fit_out):

        # get sites, scaling factor and missing noise
        sites = fit_out[ix]["layer_sites_ix"]
        gain = fit_out[ix]["gain"]
        missing_noise = fit_out[ix]["missing_noise_rms"]

        # reconstruct fitted missing noise traces
        missing_noise_traces = np.random.normal(
            0,
            missing_noise,
            [nsites, ntimepoints],
        )

        # scale traces and add missing noise
        fitted_traces[sites,:] = traces[:, sites].T * gain + missing_noise_traces[sites,:] 
        
        # release memory
        del missing_noise_traces
    return fitted_traces.T


def _scale(traces: np.array, data_conf: dict):
    """apply best fit gain to scale traces to match in vivo
    max voltage amplitude

    Returns:
        np.array: traces with amplitude and noise fitted to the in vivo
        traces
    """
    fit_out = _load_stored_fit_results(data_conf)
    traces *= fit_out[0]["gain"]
    return traces


def run(data_conf: dict, offset:bool, scale_and_add_noise: bool):
    """Rescale, and/or add missing noise and cast traces as a SpikeInterface
    RecordingExtractor

    Args:
        data_conf (dict): _description_
        offset (bool): true or false, removes each trace's mean
        scale_and_add_noise (bool): "scale_and_add_noise", "noise_20_perc_lower", "scale"
        - if true load best scaling factor and missing noise fitted
        to the in vivo traces per layer

    Returns:
        Recording:  raw SpikeInterface Recording Extractor object
    """
    # track time
    t0 = time.time()

    # set traces read path
    raw_path = data_conf["recording"]["input"]

    # get campaign parameters from one simulation
    simulation = load_campaign_params(data_conf)

    # read and cast raw trace as array (1 min/h recording)
    trace = pd.read_pickle(raw_path)
    trace = np.array(trace)

    # remove the mean (offset, 10 min/h recording)
    if offset:
        for ix in range(trace.shape[1]):
            trace[:, ix] -= np.mean(trace[:, ix])
        logger.info(f"Subtracted trace means in {np.round(time.time()-t0,2)} secs")

    # scale and add missing noise (2h:10 / h recording)
    if scale_and_add_noise == "scale_and_add_noise":
        trace = _scale_and_add_noise(trace, data_conf)
        logger.info(f"Scaled traces and added noise in {np.round(time.time()-t0,2)} secs")
    
    # scale and add missing noise (2h:10 / h recording)
    if scale_and_add_noise == "scale_adj_and_add_noise":
        trace = _scale_adj_and_add_noise(trace, data_conf)
        logger.info(f"Scaled traces with adjusted gain and added noise in {np.round(time.time()-t0,2)} secs")

    # scale and add missing noise (2h:10 / h recording)
    if isinstance(scale_and_add_noise, dict):
        trace = _scale_adj_by_and_add_noise(trace, data_conf, scale_and_add_noise)
        logger.info(f"Scaled traces with adjusted gain and added noise in {np.round(time.time()-t0,2)} secs")
                
    # scaling only
    elif scale_and_add_noise == "scale":
        trace = _scale(trace, data_conf)
        logger.info(f"Only scaled traces in {np.round(time.time()-t0,2)} secs")

    # scale and add 20% of fitted noise
    elif scale_and_add_noise == "noise_20_perc_lower":
        trace = _scale_and_add_noise_20_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 20% in {np.round(time.time()-t0,2)} secs")

    # scale and add 50% of fitted noise
    elif scale_and_add_noise == "noise_50_perc_lower":
        trace = _scale_and_add_noise_50_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 50% in {np.round(time.time()-t0,2)} secs")

    # scale and add 75% of fitted noise
    elif scale_and_add_noise == "noise_75_perc_lower":
        trace = _scale_and_add_noise_75_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 75% in {np.round(time.time()-t0,2)} secs")

    # scale and add 80% of fitted noise
    elif scale_and_add_noise == "noise_80_perc_lower":
        trace = _scale_and_add_noise_80_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 80% in {np.round(time.time()-t0,2)} secs")

    # scale and add 90% of fitted noise
    elif scale_and_add_noise == "noise_90_perc_lower":
        trace = _scale_and_add_noise_90_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 90% in {np.round(time.time()-t0,2)} secs")

    # scale and add 95% of fitted noise
    elif scale_and_add_noise == "noise_95_perc_lower":
        trace = _scale_and_add_noise_95_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 95% in {np.round(time.time()-t0,2)} secs")

    # scale and add 99% of fitted noise
    elif scale_and_add_noise == "noise_99_perc_lower":
        trace = _scale_and_add_noise_99_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 99% in {np.round(time.time()-t0,2)} secs")
    
    # scale and add 60% (40% lower) of fitted noise
    elif scale_and_add_noise == "noise_40_perc_lower":
        trace = _scale_and_add_noise_40_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 40% in {np.round(time.time()-t0,2)} secs")

    # scale and add 40% (60% lower) of fitted noise
    elif scale_and_add_noise == "noise_60_perc_lower":
        trace = _scale_and_add_noise_60_perc_lower(trace, data_conf)
        logger.info(f"Scaled traces and added noise reduced by 60% in {np.round(time.time()-t0,2)} secs")
               
    # scale only
    elif isinstance(scale_and_add_noise, (int, float)):
        trace *= scale_and_add_noise
        logger.info(f"Only scaled traces with {scale_and_add_noise} in {np.round(time.time()-t0,2)} secs")        
    else:
        NotImplementedError

    # cast trace as a SpikeInterface Recording object
    return se.NumpyRecording(
        traces_list=[trace],
        sampling_frequency=simulation["lfp_sampling_freq"],
    )


def run_from_nwb(data_conf: dict, param_conf: dict, offset:bool, scale_and_add_noise: bool):
    """Rescale, and/or add missing noise and cast traces from NWB file 
    as a SpikeInterface RecordingExtractor

    Args:
        data_conf (dict): _description_
        param_conf (dict): 
        offset (bool): true or false, removes each trace's mean
        scale_and_add_noise (bool): "scale_and_add_noise", "noise_20_perc_lower", "scale"
        - if true load best scaling factor and missing noise fitted
        to the in vivo traces per layer

    Returns:
        Recording:  raw SpikeInterface Recording Extractor object
    """
    # track time
    t0 = time.time()

    # set traces read path
    NWB_PATH = data_conf["nwb"]
    SFREQ = param_conf["sampling_freq"]
    
    # read and cast raw trace as array (1 min/h recording)
    Recording = se.NwbRecordingExtractor(NWB_PATH)
    trace = Recording.get_traces()
    
    # remove the mean (offset, 10 min/h recording)
    if offset:
        for ix in range(trace.shape[1]):
            trace[:, ix] -= np.mean(trace[:, ix])
        logger.info(f"Subtracted trace means in {np.round(time.time()-t0,2)} secs")

    # scale and add missing noise (2h:10 / h recording)
    if isinstance(scale_and_add_noise, dict):
        trace = _scale_adj_by_and_add_noise_nwb(trace, data_conf, scale_and_add_noise)
        logger.info(f"Scaled traces with adjusted gain and added noise in {np.round(time.time()-t0,2)} secs")                
    else:
        NotImplementedError

    # cast trace as a SpikeInterface Recording object
    return se.NumpyRecording(
        traces_list=[trace],
        sampling_frequency=SFREQ,
    )
    
    
def run_on_dandihub(Recording, data_conf: dict, param_conf: dict, offset:bool, scale_and_add_noise: bool):
    """Rescale, and/or add missing noise and cast traces from NWB file 
    as a SpikeInterface RecordingExtractor

    Args:
        data_conf (dict): _description_
        param_conf (dict): 
        offset (bool): true or false, removes each trace's mean
        scale_and_add_noise (bool): "scale_and_add_noise", "noise_20_perc_lower", "scale"
        - if true load best scaling factor and missing noise fitted
        to the in vivo traces per layer

    Returns:
        Recording:  raw SpikeInterface Recording Extractor object
    """
    # track time
    t0 = time.time()

    # set traces read path
    SFREQ = param_conf["sampling_freq"]
    
    # read and cast raw trace as array (1 min/h recording)
    trace = Recording.get_traces()
    
    # remove the mean (offset, 10 min/h recording)
    if offset:
        for ix in range(trace.shape[1]):
            trace[:, ix] -= np.mean(trace[:, ix])
        logger.info(f"Subtracted trace means in {np.round(time.time()-t0,2)} secs")

    # scale and add missing noise (2h:10 / h recording)
    if isinstance(scale_and_add_noise, dict):
        logger.info(f"Started scaling traces with adjusted gain and added noise ...")                
        trace = _scale_adj_by_and_add_noise_nwb(trace, data_conf, scale_and_add_noise)
        logger.info(f"Scaled traces with adjusted gain and added noise in {np.round(time.time()-t0,2)} secs")                
    else:
        NotImplementedError

    # cast trace as a SpikeInterface Recording object
    return se.NumpyRecording(
        traces_list=[trace],
        sampling_frequency=SFREQ,
    )


def load(data_conf: dict):
    """Load preprocessed recording from config

    Args:
        data_conf (dict): _description_

    Returns:
        Recording:  raw SpikeInterface Recording Extractor object
    """
    return si.load_extractor(data_conf["recording"]["output"])


def write(recording, data_conf: dict):
    """write SpikeInterface Recording object

    Args:
        recording (_type_): _description_
        data_conf (dict): _description_
    """
    # set write path
    WRITE_PATH = data_conf["recording"]["output"]

    # create write path
    shutil.rmtree(WRITE_PATH, ignore_errors=True)

    # save recording
    recording.save(folder=WRITE_PATH, format="binary", n_jobs=-1, total_memory="2G")