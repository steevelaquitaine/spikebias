# import libs
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import scipy
import spikeinterface.extractors as se 
import os
import spikeinterface as si
import spikeinterface.preprocessing as spre
import shutil 

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/preprint_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.study import amplitude
from src.nodes.prepro import preprocess
from src.nodes.dataeng.silico import recording


# SETUP PARAMETERS
GAIN = 1e5
# NOISE_STD = 3000


# horvath (vivo)
EXPERIMENT_h_vivo = "vivo_horvath"
SIMULATION_h_vivo = "probe_1" # TODO: change to probe_1
data_conf_h_vivo, param_conf_h_vivo = get_config(
    EXPERIMENT_h_vivo, SIMULATION_h_vivo
).values() 
RAW_PATH_h_vivo = data_conf_h_vivo["raw"]
PREP_PATH_h_vivo = data_conf_h_vivo["preprocessing"]["output"]["trace_file_path"]
SNR_PATH_h_vivo = data_conf_h_vivo["postprocessing"]["trace_snr"]
SNR_PLOT_DATA_h_vivo_mean = data_conf_h_vivo["postprocessing"]["snr_plot_data_mean"]
SNR_PLOT_DATA_h_vivo_ci = data_conf_h_vivo["postprocessing"]["snr_plot_data_ci"]
SNR_PLOT_DATA_h_vivo_bin = data_conf_h_vivo["postprocessing"]["snr_plot_data_bin"]
CONTACTS_h = np.arange(0,128,1)

# HORVATH (silico)
EXPERIMENT_h_silico = "silico_horvath"
SIMULATION_h_silico = "concatenated/probe_1"
data_conf_h_silico, param_conf_h_silico = get_config(
    EXPERIMENT_h_silico, SIMULATION_h_silico
).values()
RAW_PATH_h_silico = data_conf_h_silico["recording"]["output"]
PREP_PATH_h_silico = data_conf_h_silico["preprocessing"]["output"]["trace_file_path"]
SNR_PATH_h_silico = data_conf_h_silico["postprocessing"]["trace_snr"]
SNR_PLOT_DATA_h_silico_mean = data_conf_h_silico["postprocessing"]["snr_plot_data_mean"]
SNR_PLOT_DATA_h_silico_ci = data_conf_h_silico["postprocessing"]["snr_plot_data_ci"]
SNR_PLOT_DATA_h_silico_bin = data_conf_h_silico["postprocessing"]["snr_plot_data_bin"]

# calculate common bin across dataset (takes 4 mins)
N_BINS = 30


def write_npy(anr, file_write_path:str):
    parent_path = os.path.dirname(file_write_path)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
    np.save(file_write_path, anr, allow_pickle=True)


def run():

    # takes 1 secs
    # load silico
    RawRecording_h_silico = si.load_extractor(RAW_PATH_h_silico)
    PreRecording_h_silico = si.load_extractor(PREP_PATH_h_silico)

    # load vivo
    RawRecording_h_vivo = se.NwbRecordingExtractor(RAW_PATH_h_vivo)
    PreRecording_h_vivo = si.load_extractor(PREP_PATH_h_vivo)

    # takes 20 mins

    # horvath (vivo vs silico)
    anr_h_vivo = amplitude.compute_anr(PreRecording_h_vivo, CONTACTS_h)
    anr_h_silico = amplitude.compute_anr(PreRecording_h_silico, CONTACTS_h)
    write_npy(anr_h_vivo, SNR_PATH_h_vivo)
    write_npy(anr_h_silico, SNR_PATH_h_silico)

    # takes 8 min

    # load
    anr_h_vivo = np.load(SNR_PATH_h_vivo)
    anr_h_silico = np.load(SNR_PATH_h_vivo)

    anr_all = np.hstack(
        [
            np.array(anr_h_vivo).flatten(),
            np.array(anr_h_silico).flatten(),
        ]
    )
    anr_max = np.max(anr_all)
    anr_min = np.min(anr_all)
    step = (anr_max - anr_min) / N_BINS
    bins = np.arange(anr_min, anr_max + step / 2, step)

    # Compute the mean and CI of probability distributions over contacts (takes 4 min)
    # horvath (vivo)
    dist_mean_horvath, dist_ci_horvath = amplitude.compute_anr_proba_dist_stats(
        anr_h_vivo, bins
    )
    # horvath (silico)
    (
        dist_mean_silico_horvath,
        dist_ci_silico_horvath,
    ) = amplitude.compute_anr_proba_dist_stats(anr_h_silico, bins)

    # unit-test
    assert 1 - sum(dist_mean_horvath) < 1e-15, "a proba dist. should sum to 1"
    assert 1 - sum(dist_mean_silico_horvath) < 1e-15, "a proba dist. should sum to 1"

    # plot
    fig_, axis = plt.subplots(1, 1)

    # horvath (vivo)
    amplitude.plot_proba_dist_stats(
        axis,
        dist_mean_horvath[dist_mean_horvath > 0],
        dist_ci_horvath[dist_mean_horvath > 0],
        bins[:-1][dist_mean_horvath > 0],
        color=[0, 0, 0],
        ci_color=[0, 0, 0],
        label="in vivo",
        linestyle="-",
    )

    # horvath (silico)
    amplitude.plot_proba_dist_stats(
        axis,
        dist_mean_silico_horvath[dist_mean_silico_horvath > 0],
        dist_ci_horvath[dist_mean_silico_horvath > 0],
        bins[:-1][dist_mean_silico_horvath > 0],
        color=[0.9, 0, 0],
        ci_color=[0.9, 0, 0],
        label="model",
        linestyle="-",
    )

    # save plot data
    # mean
    write_npy(dist_mean_horvath[dist_mean_horvath > 0], SNR_PLOT_DATA_h_vivo_mean)
    write_npy(dist_mean_silico_horvath[dist_mean_silico_horvath > 0], SNR_PLOT_DATA_h_silico_mean)
    # ci
    write_npy(dist_ci_horvath[dist_mean_horvath > 0], SNR_PLOT_DATA_h_vivo_ci)
    write_npy(dist_ci_horvath[dist_mean_silico_horvath > 0], SNR_PLOT_DATA_h_silico_ci)    
    # bin
    write_npy(bins[:-1][dist_mean_silico_horvath > 0], SNR_PLOT_DATA_h_silico_bin)
    write_npy(bins[:-1][dist_mean_horvath > 0], SNR_PLOT_DATA_h_vivo_bin)

    # legend
    axis.set_yscale("log")
    axis.set_ylim([-1, 1.5])
    axis.set_xlim([-160, 160])
    axis.legend(frameon=False)
    axis.spines[["right", "top"]].set_visible(False)
    axis.set_ylabel("probability (ratio)")
    axis.set_xlabel("signal-to-noise ratio (a.u)")
    axis.tick_params(which="both", width=1)

    # show minor ticks
    axis.tick_params(which="major", width=1)
    # y
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8), numticks=12)
    axis.yaxis.set_major_locator(locmaj)
    axis.yaxis.set_minor_locator(locmin)
    axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

    # save
    plt.savefig("figures/2_realism/2_amplitude/snr_horvath_vs_model.pdf")
    plt.savefig("figures/2_realism/2_amplitude/snr_horvath_vs_model.svg")

run()