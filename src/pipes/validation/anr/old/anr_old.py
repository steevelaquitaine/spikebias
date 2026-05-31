"""Compute and save peak amplitude-to-noise ratio data and figure 2q

author: laquitainesteeve@gmail.com

Usage:

    # pid 3810544
    conda activate envs/spikebias

    # layer 1
    nohup python -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe1 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe1 --freq-max-horvath-probe 9999 \
        --compress-to-int16 True \
        --layer L2_3 \
        --save-data-path anrs_l2_3.npz \
        --save-fig-path fig2q_l2_3.svg \
        > out_anrs_l2_3.log 2>&1 &

    # layer 2/3 - pid: 2875961
    nohup python -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe2 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe2 --freq-max-horvath-probe 9999 \
        --compress-to-int16 True \
        --layer L2_3 \
        --save-data-path anrs_l2_3.npz \
        --save-fig-path fig2q_l2_3.svg \
        > out_anrs_l2_3.log 2>&1 &        

    # layer 4 - pid: 2919284
    nohup python -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe2 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe2 --freq-max-horvath-probe 9999 \
        --compress-to-int16 True \
        --layer L4 \
        --save-data-path anrs_l4.npz \
        --save-fig-path fig2q_l4.svg \
        > out_anrs_l4.log 2>&1 &

    # layer 5 - pid: 3091380
    # TODO: too many sites for RAM: subsample.
    sudo -S sh -c 'echo 1 > /proc/sys/vm/drop_caches'
    nohup python -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe2 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe2 --freq-max-horvath-probe 9999 \
        --recording-path-buccino dataset/00_raw/recording_buccino --freq-max-buccino 15999 \
        --compress-to-int16 True \
        --max-duration 1800\
        --layer L5 \
        --save-data-path anrs_l5.npz \
        --save-fig-path fig2q_l5.svg \
        > out_anrs_l5.log 2>&1 &        

        
    # layer 6 - pid: 3109053
    sudo -S sh -c 'echo 1 > /proc/sys/vm/drop_caches'
    nohup python -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe2 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe2 --freq-max-horvath-probe 9999 \
        --recording-path-buccino dataset/00_raw/recording_buccino --freq-max-buccino 15999 \
        --compress-to-int16 True \
        --layer L6 \
        --save-data-path anrs_l6.npz \
        --save-fig-path fig2q_l6.svg \
        > out_anrs_l6.log 2>&1 & 

Execution time: 15 minutes

Tested on: 
    
    - Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K @3.2 GHz/5.8 GHz)

Resource required:

    - CPU: multi-processing
    - 180 GB RAM
"""

# import libs
import warnings
import os
import multiprocessing
import torch
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
import logging
import logging.config
import yaml
import time
import spikeinterface as si
import spikeinterface.preprocessing as spre
import argparse

matplotlib.use("Agg")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

# move to PROJECT PATH
PROJ_PATH = "/home/steeve/steeve/epfl/code/spikebias/"
os.chdir(PROJ_PATH)

# my custom software
from src.nodes import utils
from src.nodes.validation import snr
from src.nodes.validation import amplitude as amp

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# pipeline parameters
FIG_SIZE = (1.8, 1)
N_BINS = 100
freq_min = 300

xticks = [-45, 0, 27]
yticks = [1e0, 1e-3, 1e-6]
ylims = [1e-9, 1e0]

# experiment colors
COLOR_NV = np.array([153, 153, 153]) / 255  # light gray
COLOR_NS = [0.9, 0.14, 0.15]  # red
COLOR_HV = [0.2, 0.2, 0.2]  # dark gray
COLOR_HS = np.array([26, 152, 80]) / 255  # green
COLOR_NB = [0.22, 0.5, 0.72]  # blue
COLOR_NE = [1, 0.49, 0]  # orange

# axes aesthetics
plt.rcParams["font.family"] = "Arial"
plt.rcParams["font.size"] = 6  # 5-7 with Nature neuroscience as reference
plt.rcParams["lines.linewidth"] = 0.5  # typically 0.5 - 1 pt
plt.rcParams["axes.linewidth"] = 0.5  # typically 0.5 - 1 pt
plt.rcParams["axes.spines.top"] = False
plt.rcParams["xtick.major.width"] = 0.5  # 0.8 #* 1.3
plt.rcParams["xtick.minor.width"] = 0.5  # 0.8 #* 1.3
plt.rcParams["ytick.major.width"] = 0.5  # 0.8 #* 1.3
plt.rcParams["ytick.minor.width"] = 0.5  # 0.8 #* 1.3
plt.rcParams["xtick.major.size"] = 3.5 * 1.1
plt.rcParams["xtick.minor.size"] = 2 * 1.1
plt.rcParams["ytick.major.size"] = 3.5 * 1.1
plt.rcParams["ytick.minor.size"] = 2 * 1.1
N_MAJOR_TICKS = 4
N_MINOR_TICKS = 12

# figure saving parameters
savefig_cfg = {"transparent": True, "dpi": 300, "bbox_inches": "tight"}
legend_cfg = {"frameon": False, "handletextpad": 0.1}
tight_layout_cfg = {"pad": 0.5}
LG_FRAMEON = False  # no legend frame


def get_snr_pdfs(norm_traces: np.ndarray, bins):
    """calculate amplitude-to-noise ratio distributions,
    their median and 95% confidence interval

    Args:
        norm_traces (np.ndarray): _description_
        bins (_type_): _description_
        data (list(array)): pdf by site

    Returns:
        dist_mean (np.array): 1-D array of mean snr pdf over sites
        dist_ci:  95% confidence interval of snr pdf over sites
        data (dict[list]):
        - key: "pdf_by_site"
        - value: list of 1-D arrays. One list entry per site.
    """

    # calculate average + ci of probabilities density
    # over contacts
    # calculate mean absolute deviation mad and divide
    # amplitude by mad
    proba_all = []
    for c_i in range(len(norm_traces)):
        counts, bins = np.histogram(norm_traces[c_i], bins=bins)
        proba = counts / np.sum(counts)
        proba_all.append(proba)

    # return stats
    dist_mean = np.median(np.array(proba_all), axis=0)
    # dist_mean = np.array(proba_all).mean(axis=0)
    dist_ci = 1.96 * np.std(proba_all, axis=0) / np.sqrt(len(proba_all[0]))

    # store sites' data
    data = {"pdf_by_site": proba_all}
    return dist_mean, dist_ci, data


def compute_anrs(
    layer: str,
    raw_path="dataset/00_raw/recording_marques_smith",
    freq_min=300,
    freq_max=14999,
    compress_to_int16=True,
    max_duration=1800,
):
    # track time
    t0 = time.time()

    # 1 - Load silico and vivo traces
    Recording = si.load_extractor(raw_path)
    sfreq = Recording.get_sampling_frequency()
    logger.info(f"Full recording info: {Recording}")

    # 2 - Get shorter duration (due to limited RAM)
    if max_duration < Recording.get_total_duration():
        Recording = Recording.frame_slice(start_frame=0, end_frame=max_duration * sfreq)
        logger.info(f"Shorter recording info: {Recording}")

    # 3 - get good sites (in cortex)
    # standardize layer 2_3
    # standardize sites outside
    site_ly = Recording.get_property("layers")
    site_ly = ["L2_3" if l_i == "L2" or l_i == "L3" else l_i for l_i in site_ly]
    site_ly = ["Outside" if l_i == "WM" else l_i for l_i in site_ly]

    # unit-test
    logger.info(f"Available layers in recordings are: {np.unique(site_ly)}")

    # 4 - get electrode sites in layers to keep and to drop
    sites = np.where(np.isin(site_ly, layer))[0]
    site_ids = Recording.channel_ids[sites]
    site_ids_to_remove = Recording.get_channel_ids()[
        ~np.isin(Recording.get_channel_ids(), site_ids)
    ]
    Recording = Recording.remove_channels(site_ids_to_remove)
    logger.info(f"Recording after site curation: {Recording}")
    logger.info(
        f"""Layers after site curation are: {np.unique(Recording.get_property("layers"))}"""
    )

    # 5. compress to int16 (like the Kilosort sorters)
    # nov 22 2024
    if compress_to_int16:
        Recording = spre.astype(Recording, "int16")
        logger.info(f"Compressed to int16 in {np.round(time.time()-t0,2)} secs")
    logger.info(f"Recording dtype is {Recording.dtype}")

    # 6. preprocess
    # bandpass filter
    logger.info(f"Bandpass filtering ....")
    Recording = spre.bandpass_filter(
        recording=Recording, freq_min=freq_min, freq_max=freq_max
    )
    logger.info(f"Done bandpass filtering.")

    # apply referencing
    logger.info(f"Common referencing ....")
    Recording = spre.common_reference(Recording, reference="global", operator="median")
    logger.info("Done common referencing")
    logger.info(f"Done preprocessing: {Recording}")

    # 7. load curated preprocessed traces
    traces = Recording.get_traces()
    logger.info(f"Loaded traces in {np.round(time.time()-t0,2)} secs")

    # 8. compute snrs and save
    logger.info(f"Computing ANR ...")
    anr = snr.get_snrs_parallel(traces).astype(np.float32)
    logger.info(f"Done computing ANR in {np.round(time.time()-t0,2)} secs")
    return anr


def compute_anrs_buccino(
    raw_path="dataset/00_raw/recording_buccino",
    freq_min=300,
    freq_max=14999,
    compress_to_int16=True,
    max_duration=1800,
):
    # track time
    t0 = time.time()

    # keep 20 channels
    # (more require too much ram)
    site_ids_to_remove = np.arange(20, 384, 1)

    # 1 - Load silico and vivo traces
    Recording = si.load_extractor(raw_path)
    sfreq = Recording.get_sampling_frequency()
    logger.info(f"Full recording info: {Recording}")

    # 2 - Get shorter duration (due to limited RAM)
    if max_duration < Recording.get_total_duration():
        Recording = Recording.frame_slice(start_frame=0, end_frame=max_duration * sfreq)
        logger.info(f"Shorter recording info: {Recording}")

    # 3 - curate sites
    Recording = Recording.remove_channels(site_ids_to_remove)
    logger.info(f"""L5 site ids after curation are: {Recording.get_channel_ids()}""")

    # 4. compress to int16 (like the Kilosort sorters)
    # nov 22 2024
    if compress_to_int16:
        Recording = spre.astype(Recording, "int16")
        logger.info(f"Compressed to int16 in {np.round(time.time()-t0,2)} secs")
    logger.info(f"Recording dtype is {Recording.dtype}")

    # 5. preprocess
    # bandpass filter
    logger.info(f"Bandpass filtering ....")
    Recording = spre.bandpass_filter(
        recording=Recording, freq_min=freq_min, freq_max=freq_max
    )
    logger.info(f"Done bandpass filtering.")

    # apply referencing
    logger.info(f"Common referencing ....")
    Recording = spre.common_reference(Recording, reference="global", operator="median")
    logger.info("Done common referencing")
    logger.info(f"Done preprocessing: {Recording}")

    # 6. load curated preprocessed traces
    traces = Recording.get_traces()
    logger.info(f"Loaded traces in {np.round(time.time()-t0,2)} secs")

    # 7. compute snrs and save
    logger.info(f"Computing ANR ...")
    anr = snr.get_snrs_parallel(traces).astype(np.float32)
    logger.info(f"Done computing ANR in {np.round(time.time()-t0,2)} secs")
    return anr


if __name__ == "__main__":
    """Entry point"""

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compute psds")

    # recording paths
    parser.add_argument(
        "--recording-path-marques",
        default="./dataset/00_raw/recording_marques_smith",
        help="recording path",
    )
    parser.add_argument(
        "--recording-path-npx-spont",
        default="./dataset/00_raw/recording_npx_spont",
        help="recording path",
    )
    parser.add_argument(
        "--recording-path-npx-evoked",
        default="./dataset/00_raw/recording_npx_evoked",
        help="recording path",
    )
    parser.add_argument(
        "--recording-path-dense-probe",
        default="./dataset/00_raw/recording_dense_probe1",
        help="recording path",
    )
    parser.add_argument(
        "--recording-path-horvath-probe",
        default="./dataset/00_raw/recording_horvath_probe1",
        help="recording path",
    )
    parser.add_argument(
        "--recording-path-buccino",
        default="./dataset/00_raw/recording_buccino",
        help="recording path",
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=1800,
        help="max duration of recording",
    )
    # data compression to int16
    parser.add_argument(
        "--compress-to-int16",
        type=bool,
        default=False,
        help="recording compression to int16",
    )
    # max frequence for high-pass filtering
    parser.add_argument(
        "--freq-max-marques",
        type=int,
        default=None,
        help="freq max of pass filter cutoff",
    )
    parser.add_argument(
        "--freq-max-npx-spont",
        type=int,
        default=None,
        help="freq max of pass filter cutoff",
    )
    parser.add_argument(
        "--freq-max-npx-evoked",
        type=int,
        default=None,
        help="freq max of pass filter cutoff",
    )
    parser.add_argument(
        "--freq-max-dense-probe",
        type=int,
        default=None,
        help="freq max of pass filter cutoff",
    )
    parser.add_argument(
        "--freq-max-horvath-probe",
        type=int,
        default=None,
        help="freq max of pass filter cutoff",
    )
    parser.add_argument(
        "--freq-max-buccino",
        type=int,
        default=None,
        help="freq max of pass filter cutoff",
    )
    # layer
    parser.add_argument(
        "--layer",
        default=None,
        help="cortical layer",
    )
    # saving path
    parser.add_argument(
        "--save-data-path",
        default="./dataset/01_intermediate/anrs/anrs.npz",
        help="save path for amplitude to noise ratio data",
    )
    parser.add_argument(
        "--save-fig-path",
        default="./figures/08_source_data/fig2q/fig2q.svg",
        help="save fig of amplitude-to-noise-ratio distribution",
    )
    args = parser.parse_args()

    # report parameters for visual check
    logger.info(f"layer: {args.layer}")
    logger.info(f"max_duration: {args.max_duration}")
    logger.info(f"freq_max: {args.freq_max_marques}")
    logger.info(f"freq_max: {args.freq_max_npx_spont}")
    logger.info(f"freq_max: {args.freq_max_npx_evoked}")
    logger.info(f"freq_max: {args.freq_max_dense_probe}")
    logger.info(f"freq_max: {args.freq_max_horvath_probe}")
    logger.info(f"freq_max: {args.freq_max_buccino}")

    # run ---------------------------------------------

    # track time
    t_start = time.time()

    # get specs
    print("available cpus:", multiprocessing.cpu_count())
    print("available gpus:", torch.cuda.is_available())

    # compute anrs
    anrs_m = compute_anrs(
        layer=args.layer,
        raw_path=args.recording_path_marques,
        freq_max=args.freq_max_marques,
        compress_to_int16=args.compress_to_int16,
        max_duration=args.max_duration,
    )
    logger.info(f"Done computing anr for marques-smith ----------")

    anrs_ns = compute_anrs(
        layer=args.layer,
        raw_path=args.recording_path_npx_spont,
        freq_max=args.freq_max_npx_spont,
        compress_to_int16=args.compress_to_int16,
        max_duration=args.max_duration,
    )
    logger.info(f"Done computing anr for npx_spont ----------")

    anrs_ne = compute_anrs(
        layer=args.layer,
        raw_path=args.recording_path_npx_evoked,
        freq_max=args.freq_max_npx_evoked,
        compress_to_int16=args.compress_to_int16,
        max_duration=args.max_duration,
    )
    logger.info(f"Done computing anr for npx_evoked ----------")

    anrs_d1 = compute_anrs(
        layer=args.layer,
        raw_path=args.recording_path_dense_probe,
        freq_max=args.freq_max_dense_probe,
        compress_to_int16=args.compress_to_int16,
        max_duration=args.max_duration,
    )
    logger.info(f"Done computing anr for {args.recording_path_dense_probe} ----------")

    anrs_h1 = compute_anrs(
        layer=args.layer,
        raw_path=args.recording_path_horvath_probe,
        freq_max=args.freq_max_horvath_probe,
        compress_to_int16=args.compress_to_int16,
        max_duration=args.max_duration,
    )
    logger.info(
        f"Done computing anr for {args.recording_path_horvath_probe} ----------"
    )

    # when layer is L5, include Buccino's model
    if args.layer == "L5":

        anrs_b = compute_anrs_buccino(
            raw_path=args.recording_path_buccino,
            freq_max=args.freq_max_buccino,
            compress_to_int16=args.compress_to_int16,
            max_duration=args.max_duration,
        )
        logger.info(f"Done computing anr for {args.recording_path_buccino} ----------")

    else:

        anrs_b = np.nan

    # get the common bins
    min_anr = np.nanmin(
        np.array(
            [
                np.min(anrs_m),
                np.min(anrs_ns),
                np.min(anrs_ne),
                np.min(anrs_d1),
                np.min(anrs_h1),
                np.min(anrs_b),
            ]
        )
    )
    max_anr = np.nanmax(
        np.array(
            [
                np.max(anrs_m),
                np.max(anrs_ns),
                np.max(anrs_ne),
                np.max(anrs_d1),
                np.max(anrs_h1),
                np.max(anrs_b),
            ]
        )
    )

    # report min and max ANRs
    logger.info(f"ANR min-max: {min_anr} - {max_anr}")

    # get the common ANR bins across all experiments
    steps = (max_anr - min_anr) / N_BINS

    bins = np.arange(min_anr, max_anr + steps / 2, steps)
    logger.info("Done computing bins across experiments.")

    # track time
    t0 = time.time()

    # Compute the summary statistics of the ANRs
    logger.info(f"Computing anr stats...")
    mean_m, ci_m, _ = amp.get_snr_pdfs(anrs_m, bins)
    mean_ns, ci_ns, _ = amp.get_snr_pdfs(anrs_ns, bins)
    mean_ne, ci_ne, _ = amp.get_snr_pdfs(anrs_ne, bins)
    mean_d1, ci_d1, _ = amp.get_snr_pdfs(anrs_d1, bins)
    mean_h1, ci_h1, _ = amp.get_snr_pdfs(anrs_h1, bins)

    if args.layer == "L5":
        mean_b, ci_b, _ = amp.get_snr_pdfs(anrs_b, bins)

    logger.info(f"Done computing ANR stats in {np.round(time.time() - t0,2)} secs")

    # save the ANR data
    np.savez(
        args.save_data_path,
        anrs_m=anrs_m,
        anrs_ns=anrs_ns,
        anrs_ne=anrs_ne,
        anrs_d1=anrs_d1,
        anrs_h1=anrs_h1,
        anrs_b=anrs_b,
    )
    logger.info(f"All completed in {np.round(time.time()-t_start,2)} secs")

    # set parameters
    pm = {
        "linestyle": "-",
        "linewidth": 1,
        "marker": "None",
    }

    FIG_SIZE = (1.9, 1.5)

    # plot
    fig, ax = plt.subplots(1, 1, figsize=FIG_SIZE)

    if not args.layer == "L5":
        # neuropixels
        ax = amp.plot_snr_pdf_all(
            ax,
            mean_m,
            mean_ns,
            mean_ne,
            ci_m,
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
            mean_h1,
            mean_d1,
            [0],
            ci_h1,
            ci_d1,
            [0],
            bins,
            COLOR_HV,
            COLOR_HS,
            [0],
            pm,
        )
    else:
        amp.plot_anr_pdf_l5(
            ax,
            mean_m,
            mean_ns,
            mean_ne,
            mean_b,
            ci_m,
            ci_ns,
            ci_ne,
            ci_b,
            bins,
            COLOR_NV,
            COLOR_NS,
            COLOR_NE,
            COLOR_NB,
            pm,
        )
        amp.plot_anr_pdf_l5(
            ax,
            mean_h1,
            mean_d1,
            [0],
            [0],
            ci_h1,
            ci_d1,
            [0],
            [0],
            bins,
            COLOR_HV,
            COLOR_HS,
            [0],
            [0],
            pm,
        )

    xmin, xmax = ax.get_xlim()

    # set x and y axis ticks and limits
    # ax.set_xlim([np.floor(xmin), np.ceil(xmax)])
    ax.set_xticks(
        xticks,
        xticks,
    )
    ax.set_xlim([xticks[0], xticks[-1]])

    # set yticks and labels
    labels = [f"$10^{{{int(np.log10(t))}}}$" for t in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(ylims)

    # tighten
    fig.tight_layout(**tight_layout_cfg)

    # save
    plt.savefig(args.save_fig_path, **savefig_cfg)
    logger.info("Saved ANR plot.")
