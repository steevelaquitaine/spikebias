"""Compute and save peak amplitude-to-noise ratio

author: laquitainesteeve@gmail.com

Usage:

    # pid 3675220
    conda activate envs/spikebias
    nohup python -m src.pipes.validation.anr.anr_l1 > out_anrs_l1.log 2>&1 &

    
    nohup python -m src.pipes.validation.anr.anr_l1 \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe1 dataset/00_raw/recording_dense_probe1 --freq-max-dense-probe1 9999 \
        --recording-path-horvath-probe1 dataset/00_raw/recording_horvath_probe1 --freq-max-horvath-probe1 9999 \
        --layer L1 \
        --save-data-path anrs.npz \
        --save-fig-path fig2q_l1.svg \
        > out_anrs_l1.log 2>&1 &


Execution time: 12 minutes
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
lyrs = ["L1"]
N_BINS = 100
freq_min = 300
min_anr = -20  # as in paper
max_anr = 17  # as in paper


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
savefig_cfg = {"transparent": True, "dpi": 300}
legend_cfg = {"frameon": False, "handletextpad": 0.1}
tight_layout_cfg = {"pad": 0.5}
LG_FRAMEON = False  # no legend frame


def compute_anrs(
    raw_path="dataset/00_raw/recording_marques_smith", freq_min=300, freq_max=14999
):

    # track time
    t0 = time.time()

    # 1 - Load silico and vivo traces
    Recording = si.load_extractor(raw_path)

    # 2 - get good sites (in cortex)
    site_ly = Recording.get_property("layers")
    sites = np.where(np.isin(site_ly, lyrs))[0]
    site_ids = Recording.channel_ids[sites]
    site_ids_to_remove = Recording.get_channel_ids()[
        ~np.isin(Recording.get_channel_ids(), site_ids)
    ]
    Recording = Recording.remove_channels(site_ids_to_remove)
    logger.info(f"Recording after site curation: {Recording}")

    # 3. compress to int16 (like the Kilosort sorters)
    # nov 22 2024
    Recording = spre.astype(Recording, "int16")
    logger.info(f"Compressed to int16 in {np.round(time.time()-t0,2)} secs")

    # 4. preprocess
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

    # 5. load curated preprocessed traces
    traces = Recording.get_traces()
    logger.info(f"Loaded traces in {np.round(time.time()-t0,2)} secs")

    # 6. compute snrs and save
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
        "--recording-path-dense-probe1",
        default="./dataset/00_raw/recording_dense_probe1",
        help="recording path",
    )
    parser.add_argument(
        "--recording-path-horvath-probe1",
        default="./dataset/00_raw/recording_horvath_probe1",
        help="recording path",
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
        "--freq-max-dense-probe1",
        type=int,
        default=None,
        help="freq max of pass filter cutoff",
    )
    parser.add_argument(
        "--freq-max-horvath-probe1",
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
    # parser.add_argument(
    #     "--gain-to-uv", type=float, default=1, help="gain to uV conversion factor"
    # )
    # parser.add_argument(
    #     "--preprocess",
    #     type=bool,
    #     default=False,
    #     help="apply highpass filtering and common referencing",
    # )
    # parser.add_argument(
    #     "--duration",
    #     type=int,
    #     default=3600,
    #     help="max recording duration in seconds for preprocessing (cheaper)",
    # )
    # parser.add_argument(
    #     "--bandpass-filter", type=bool, default=False, help="bandpass filtering"
    # )
    # parser.add_argument(
    #     "--freq-min", type=int, default=300, help="high pass filter cutoff"
    # )
    # parser.add_argument(
    #     "--freq-max", type=int, default=None, help="freq max of pass filter cutoff"
    # )
    # parser.add_argument("--layers", nargs="+", help="list of layers to analyse")
    # parser.add_argument(
    #     "--keep-first-n-sites",
    #     type=int,
    #     help="number of sites to keep from the first to N-th site",
    # )
    # parser.add_argument(
    #     "--keep-last-n-sites",
    #     type=int,
    #     help="number of sites to keep from the last N-th site",
    # )
    # parser.add_argument(
    #     "--filter_window", type=str, default="hann", help="welch psd filter window"
    # )
    args = parser.parse_args()

    # report parameters for visual check
    logger.info(f"freq_max: {args.freq_max_marques}")
    logger.info(f"freq_max: {args.freq_max_npx_spont}")
    logger.info(f"freq_max: {args.freq_max_npx_evoked}")
    logger.info(f"freq_max: {args.freq_max_dense_probe1}")
    logger.info(f"freq_max: {args.freq_max_horvath_probe1}")
    logger.info(f"layer: {args.layer}")

    # run ---------------------------------------------

    # track time
    t_start = time.time()

    print("available cpus:", multiprocessing.cpu_count())
    print("available gpus:", torch.cuda.is_available())

    # compute anrs
    anrs_m = compute_anrs(
        raw_path=args.recording_path_marques, freq_max=args.freq_max_marques
    )
    logger.info(f"Done computing anr for marques-smith.")

    anrs_ns = compute_anrs(
        raw_path=args.recording_path_npx_spont, freq_max=args.freq_max_npx_spont
    )
    logger.info(f"Done computing anr for npx_spont.")

    anrs_ne = compute_anrs(
        raw_path=args.recording_path_npx_evoked,
        freq_max=args.freq_max_freq_max_npx_evoked,
    )
    logger.info(f"Done computing anr for npx_evoked.")

    anrs_d1 = compute_anrs(
        raw_path=args.recording_path_dense_probe1,
        freq_max=args.freq_max_freq_max_dense_probe1,
    )
    logger.info(f"Done computing anr for dense_probe1.")

    anrs_h1 = compute_anrs(
        raw_path=args.recording_path_horvath_probe1,
        freq_max=args.freq_max_freq_max_horvath_probe1,
    )
    logger.info(f"Done computing anr for horvath_probe1.")

    # get the common bins
    min_anr = np.array(
        [
            np.min(anrs_m),
            np.min(anrs_ns),
            np.min(anrs_ne),
            np.min(anrs_d1),
            np.min(anrs_h1),
        ]
    ).min()
    max_anr = np.array(
        [
            np.max(anrs_m),
            np.max(anrs_ns),
            np.max(anrs_ne),
            np.max(anrs_d1),
            np.max(anrs_h1),
        ]
    ).max()
    print(max_anr)

    # get the common ANR bins across all experiments
    steps = (max_anr - min_anr) / N_BINS

    bins = np.arange(min_anr, max_anr + steps / 2, steps)
    logger.info("Done computing bins across experiments.")

    # track time
    t0 = time.time()

    # Compute the summary statistics of the ANRs
    mean_m, ci_m, _ = amp.get_snr_pdfs(anrs_m, bins)
    mean_ns, ci_ns, _ = amp.get_snr_pdfs(anrs_ns, bins)
    mean_ne, ci_ne, _ = amp.get_snr_pdfs(anrs_ne, bins)
    mean_d1, ci_d1, _ = amp.get_snr_pdfs(anrs_d1, bins)
    mean_h1, ci_h1, _ = amp.get_snr_pdfs(anrs_h1, bins)
    logger.info(f"Done computing ANR stats in {np.round(time.time() - t0,2)} secs")

    # save the ANR data
    np.savez(
        # os.path.join(PROJ_PATH, "anrs.npz"),
        args.save_data_path,
        anrs=anrs_m,
        anrs_ns=anrs_ns,
        anrs_ne=anrs_ne,
        anrs_d1=anrs_d1,
        anrs_h1=anrs_h1,
    )
    logger.info(f"All completed in {np.round(time.time()-t_start,2)} secs")

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
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(
        [np.floor(xmin).astype(int), 0, np.ceil(xmax).astype(int)],
        [np.floor(xmin).astype(int), 0, np.ceil(xmax).astype(int)],
    )
    ax.set_xlim([np.floor(xmin), np.ceil(xmax)])

    # tighten
    fig.tight_layout(**tight_layout_cfg)

    # save
    # plt.savefig(os.path.join(PROJ_PATH, "fig2q_l1.svg"), **savefig_cfg)
    plt.savefig(args.save_fig_path, **savefig_cfg)
    logger.info("Saved ANR plot.")
