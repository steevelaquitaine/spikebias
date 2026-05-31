"""Compute and save peak amplitude-to-noise ratio data and figure 2q

author: laquitainesteeve@gmail.com

Usage:
    
    # activate conda environment 
    conda activate envs/spikebias

    # layer 1 - 3590447
    nohup bash -c 'python -u -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe1 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe1 --freq-max-horvath-probe 9999 \
        --freq-min 300 \
        --compress-to-int16 True \
        --max-duration 600 \
        --layer L1 \
        --save-data-path figures/8_source_data/fig2q/fig2q_l1 \
        --save-fig-path figures/8_source_data/fig2q/fig2q_l1.svg \
        --xlim -20 17 \
        > logs/fig2q/out_anrs_l1.log 2>&1 && curl -d "Done anr-L1" ntfy.sh/Code || curl -d "❌ Crash anr-L1" ntfy.sh/Code' \
        > /dev/null 2>&1 &

    # layer 2/3 - 2402020
    # note: from probe 1
    nohup bash -c 'python -u -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe1 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe1 --freq-max-horvath-probe 9999 \
        --freq-min 300 \
        --compress-to-int16 True \
        --max-duration 600 \
        --layer L2_3 \
        --save-data-path figures/8_source_data/fig2q/fig2q_l23 \
        --save-fig-path figures/8_source_data/fig2q/fig2q_l23.svg \
        --xlim -69 36 \
        > logs/fig2q/out_anrs_l23.log 2>&1 && curl -d "Done anr-L23" ntfy.sh/Code || curl -d "❌ Crash anr-L23" ntfy.sh/Code' \
        > /dev/null 2>&1 &        
    
    # layer 4 - 2403680
    # note: from probe 2
    nohup bash -c 'python -u -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe2 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe2 --freq-max-horvath-probe 9999 \
        --freq-min 300 \
        --compress-to-int16 True \
        --max-duration 600 \
        --layer L4 \
        --save-data-path figures/8_source_data/fig2q/fig2q_l4 \
        --save-fig-path figures/8_source_data/fig2q/fig2q_l4.svg \
        --xlim -41 38 \
        > logs/fig2q/out_anrs_l4.log 2>&1 && curl -d "Done anr-L4" ntfy.sh/Code || curl -d "❌ Crash anr-L4" ntfy.sh/Code' \
        > /dev/null 2>&1 &     

    # layer 5 - 2406029
    nohup bash -c 'python -u -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe2 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe2 --freq-max-horvath-probe 9999 \
        --recording-path-buccino dataset/00_raw/recording_buccino --freq-max-buccino 15999 \
        --freq-min 300 \
        --compress-to-int16 True \
        --max-duration 600 \
        --layer L5 \
        --save-data-path figures/8_source_data/fig2q/fig2q_l5 \
        --save-fig-path figures/8_source_data/fig2q/fig2q_l5.svg \
        --xlim -45 27 \
        > logs/fig2q/out_anrs_l5.log 2>&1 && curl -d "Done anr-L5" ntfy.sh/Code || curl -d "❌ Crash anr-L5" ntfy.sh/Code' \
        > /dev/null 2>&1 &

    # layer 6 - 2408449
    nohup bash -c 'python -u -m src.pipes.validation.anr.anr \
        --recording-path-marques dataset/00_raw/recording_marques_smith --freq-max-marques 14999 \
        --recording-path-npx-spont dataset/00_raw/recording_npx_spont --freq-max-npx-spont 19999 \
        --recording-path-npx-evoked dataset/00_raw/recording_npx_evoked --freq-max-npx-evoked 9999 \
        --recording-path-dense-probe dataset/00_raw/recording_dense_probe3 --freq-max-dense-probe 9999 \
        --recording-path-horvath-probe dataset/00_raw/recording_horvath_probe3 --freq-max-horvath-probe 9999 \
        --recording-path-buccino dataset/00_raw/recording_buccino --freq-max-buccino 15999 \
        --freq-min 300 \
        --compress-to-int16 True \
        --max-duration 600 \
        --layer L6 \
        --save-data-path figures/8_source_data/fig2q/fig2q_l6 \
        --save-fig-path figures/8_source_data/fig2q/fig2q_l6.svg \
        --xlim -31 31 \
        > logs/fig2q/out_anrs_l6.log 2>&1 && curl -d "Done anr-L6" ntfy.sh/Code || curl -d "❌ Crash anr-L6" ntfy.sh/Code' \
        > /dev/null 2>&1 &

Execution time: 

    ~15 min (3 passes over each recording, chunk_samples=500_000)

Tested on:

    - Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K @3.2 GHz/5.8 GHz)

Resource required:
    - CPU: multi-processing
    - RAM: ~4 MB per chunk (chunk_sites=10, chunk_samples=6_000_000)
    - PEAK RAM of 10 GB
"""

# import libs
import warnings
import os
import gc
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
from memory_profiler import memory_usage
import requests
import traceback

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
N_BINS = 100
CHUNK_SITES = 384  # process all sites at once per time chunk
CHUNK_SAMPLES = 6_000_000  # number of time samples per chunk (~3s at 32kHz)

xticks = [-20, 0, 17]
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
plt.rcParams["font.size"] = 6
plt.rcParams["lines.linewidth"] = 0.5
plt.rcParams["axes.linewidth"] = 0.5
plt.rcParams["axes.spines.top"] = False
plt.rcParams["xtick.major.width"] = 0.5
plt.rcParams["xtick.minor.width"] = 0.5
plt.rcParams["ytick.major.width"] = 0.5
plt.rcParams["ytick.minor.width"] = 0.5
plt.rcParams["xtick.major.size"] = 3.5 * 1.1
plt.rcParams["xtick.minor.size"] = 2 * 1.1
plt.rcParams["ytick.major.size"] = 3.5 * 1.1
plt.rcParams["ytick.minor.size"] = 2 * 1.1

# figure saving parameters
savefig_cfg = {"transparent": True, "dpi": 300, "bbox_inches": "tight"}
legend_cfg = {"frameon": False, "handletextpad": 0.1}
tight_layout_cfg = {"pad": 0.5}

# =============================================================================
# Notification
# =============================================================================


def notify(msg):
    requests.post("https://ntfy.sh/Code", data=msg.encode("utf-8"))


# =============================================================================
# Recording loading and preprocessing
# =============================================================================


def load_and_preprocess(
    raw_path: str,
    freq_min: int,
    freq_max: int,
    compress_to_int16: bool,
    max_duration: int,
    layer: str = None,
    is_buccino: bool = False,
    n_buccino_sites: int = 20,
):
    """Load and preprocess a recording. No traces loaded into RAM.

    Args:
        raw_path: path to the recording
        freq_min: bandpass filter min frequency
        freq_max: bandpass filter max frequency
        compress_to_int16: whether to compress to int16
        max_duration: max duration in seconds
        layer: cortical layer to keep (None for buccino)
        is_buccino: whether this is the buccino recording
        n_buccino_sites: number of sites to keep for buccino

    Returns:
        Recording: preprocessed SpikeInterface recording object
    """
    t0 = time.time()

    Recording = si.load_extractor(raw_path)
    sfreq = Recording.get_sampling_frequency()
    logger.info(f"Loaded recording: {Recording}")

    # trim duration
    if max_duration < Recording.get_total_duration():
        Recording = Recording.frame_slice(
            start_frame=0, end_frame=int(max_duration * sfreq)
        )
        logger.info(f"Trimmed recording: {Recording}")

    # select sites
    if is_buccino:
        site_ids_to_remove = Recording.get_channel_ids()[n_buccino_sites:]
        Recording = Recording.remove_channels(site_ids_to_remove)
        logger.info(f"Buccino sites kept: {Recording.get_channel_ids()}")
    else:
        site_ly = Recording.get_property("layers")
        site_ly = ["L2_3" if l_i in ("L2", "L3") else l_i for l_i in site_ly]
        site_ly = ["Outside" if l_i == "WM" else l_i for l_i in site_ly]
        logger.info(f"Available layers: {np.unique(site_ly)}")

        sites = np.where(np.isin(site_ly, layer))[0]
        site_ids = Recording.channel_ids[sites]
        site_ids_to_remove = Recording.get_channel_ids()[
            ~np.isin(Recording.get_channel_ids(), site_ids)
        ]
        Recording = Recording.remove_channels(site_ids_to_remove)
        logger.info(f"Recording after layer curation: {Recording}")
        logger.info(f"""Layers after curation: {Recording.get_property("layers")}""")

    # compress
    if compress_to_int16:
        Recording = spre.astype(Recording, "int16")

    # bandpass filter
    Recording = spre.bandpass_filter(
        recording=Recording, freq_min=freq_min, freq_max=freq_max
    )

    # common reference
    Recording = spre.common_reference(Recording, reference="global", operator="median")

    logger.info(f"Preprocessing done in {np.round(time.time()-t0, 2)} secs")
    return Recording


# =============================================================================
# Pass 0: compute full-recording MAD per site (one site at a time)
# =============================================================================


def compute_mad_by_site(Recording, chunk_samples=CHUNK_SAMPLES):
    """Two-pass chunk-based MAD. Never loads more than
    chunk_sites x chunk_samples into RAM.
    Peak RAM = CHUNK_SITES x chunk_samples x 4 bytes ~ 153 MB.
    """
    all_channel_ids = Recording.get_channel_ids()
    n_sites = len(all_channel_ids)
    n_samples = Recording.get_num_samples()
    mads = np.zeros(n_sites, dtype=np.float32)

    # Pass A: compute mean per site over full recording
    # accumulate sum and count chunk by chunk
    sums = np.zeros(n_sites, dtype=np.float64)
    counts = np.zeros(n_sites, dtype=np.int64)

    for t_start in range(0, n_samples, chunk_samples):
        t_end = min(t_start + chunk_samples, n_samples)
        # load ALL sites at once for this time chunk
        traces = Recording.get_traces(start_frame=t_start, end_frame=t_end).astype(
            np.float32
        )  # shape: (chunk_samples, n_sites)
        sums += traces.sum(axis=0)
        counts += traces.shape[0]
        del traces
        gc.collect()

    means = (sums / counts).astype(np.float32)  # shape: (n_sites,) — tiny
    logger.info(f"  Pass A done: means computed for {n_sites} sites")

    # Pass B: compute MAD per site using precomputed means
    # accumulate sum of absolute deviations chunk by chunk
    mad_sums = np.zeros(n_sites, dtype=np.float64)

    for t_start in range(0, n_samples, chunk_samples):
        t_end = min(t_start + chunk_samples, n_samples)
        traces = Recording.get_traces(start_frame=t_start, end_frame=t_end).astype(
            np.float32
        )  # shape: (chunk_samples, n_sites)
        mad_sums += np.abs(traces - means).sum(axis=0)
        del traces
        gc.collect()

    mads = (mad_sums / counts).astype(np.float32)
    mads[mads == 0] = 1.0
    logger.info(f"  Pass B done: MADs computed for {n_sites} sites")
    return mads


# =============================================================================
# Pass 1: get global ANR range across all sites and time
# =============================================================================


def get_anr_range(
    Recording,
    mads: np.ndarray,
    chunk_sites: int = CHUNK_SITES,
    chunk_samples: int = CHUNK_SAMPLES,
):
    """Compute global ANR min/max using full-recording MADs.
    Peak RAM = chunk_sites x chunk_samples x 4 bytes ~ 4 MB.

    Args:
        Recording: preprocessed SpikeInterface recording
        mads: full-recording MAD per site, shape (n_sites,)
        chunk_sites: number of sites per spatial chunk
        chunk_samples: number of time samples per chunk

    Returns:
        global_min, global_max: ANR range across all sites and time
    """
    all_channel_ids = Recording.get_channel_ids()
    n_sites = len(all_channel_ids)
    n_samples = Recording.get_num_samples()
    global_min, global_max = np.inf, -np.inf

    for i in range(0, n_sites, chunk_sites):
        chunk_ids = all_channel_ids[i : i + chunk_sites]
        chunk_mads = mads[i : i + chunk_sites]  # shape: (chunk_sites,)
        rec_chunk = Recording.channel_slice(channel_ids=chunk_ids)

        for t_start in range(0, n_samples, chunk_samples):
            t_end = min(t_start + chunk_samples, n_samples)
            traces = rec_chunk.get_traces(start_frame=t_start, end_frame=t_end).astype(
                np.float32
            )  # shape: (chunk_samples, chunk_sites)

            # normalise using full-recording MAD — identical to original
            traces /= chunk_mads  # in-place, no copy

            global_min = min(global_min, float(traces.min()))
            global_max = max(global_max, float(traces.max()))
            del traces

        gc.collect()

    return global_min, global_max


# =============================================================================
# Pass 2: compute per-site histogram counts
# =============================================================================


def compute_histogram_counts_by_chunk(
    Recording,
    mads: np.ndarray,
    bins: np.ndarray,
    chunk_sites: int = CHUNK_SITES,
    chunk_samples: int = CHUNK_SAMPLES,
):
    """Compute per-site ANR histogram counts using full-recording MADs.
    Peak RAM = chunk_sites x chunk_samples x 4 bytes ~ 4 MB.

    Args:
        Recording: preprocessed SpikeInterface recording
        mads: full-recording MAD per site, shape (n_sites,)
        bins: histogram bin edges (common across all recordings)
        chunk_sites: number of sites per spatial chunk
        chunk_samples: number of time samples per chunk

    Returns:
        counts_per_site: list of 1-D int64 arrays, one per site
    """
    all_channel_ids = Recording.get_channel_ids()
    n_sites = len(all_channel_ids)
    n_samples = Recording.get_num_samples()
    n_bins = len(bins) - 1

    counts_per_site = []

    for i in range(0, n_sites, chunk_sites):
        chunk_ids = all_channel_ids[i : i + chunk_sites]
        n_chunk_sites = len(chunk_ids)
        chunk_mads = mads[i : i + chunk_sites]  # shape: (chunk_sites,)
        rec_chunk = Recording.channel_slice(channel_ids=chunk_ids)

        # accumulate histogram counts over time for this site chunk
        site_counts = np.zeros((n_chunk_sites, n_bins), dtype=np.int64)

        for t_start in range(0, n_samples, chunk_samples):
            t_end = min(t_start + chunk_samples, n_samples)
            traces = rec_chunk.get_traces(start_frame=t_start, end_frame=t_end).astype(
                np.float32
            )  # shape: (chunk_samples, chunk_sites)

            # normalise using full-recording MAD — identical to original
            traces /= chunk_mads  # in-place, no copy

            # accumulate histogram per site
            for s in range(n_chunk_sites):
                c, _ = np.histogram(traces[:, s], bins=bins)
                site_counts[s] += c

            del traces
            gc.collect()

        counts_per_site.extend(list(site_counts))
        logger.info(f"  Sites {i}-{min(i+chunk_sites, n_sites)}/{n_sites} done")

    return counts_per_site


# =============================================================================
# Pass 3: compute PDF stats from histogram counts
# =============================================================================


def counts_to_pdf(counts_per_site: list):
    """Convert per-site histogram counts to median PDF and 95% CI.
    Numerically identical to original get_snr_pdfs().

    Args:
        counts_per_site: list of 1-D int64 arrays, one per site

    Returns:
        dist_mean: median PDF across sites
        dist_ci: 95% confidence interval
    """
    proba_all = []
    for counts in counts_per_site:
        total = counts.sum()
        # if total > 0:
        proba_all.append(counts / total)

    proba_arr = np.array(proba_all)
    dist_mean = np.median(proba_arr, axis=0)
    dist_ci = 1.96 * np.std(proba_arr, axis=0) / np.sqrt(proba_arr.shape[1])
    return dist_mean, dist_ci


# =============================================================================
# Main pipeline
# =============================================================================


def main(args):
    """Run main pipeline.

    Args:
        args: parsed pipeline arguments
    """
    # report parameters
    logger.info(f"layer: {args.layer}")
    logger.info(f"max_duration: {args.max_duration}")
    logger.info(f"freq_min: {args.freq_min}")
    logger.info(f"freq_max_marques: {args.freq_max_marques}")
    logger.info(f"freq_max_npx_spont: {args.freq_max_npx_spont}")
    logger.info(f"freq_max_npx_evoked: {args.freq_max_npx_evoked}")
    logger.info(f"freq_max_dense_probe: {args.freq_max_dense_probe}")
    logger.info(f"freq_max_horvath_probe: {args.freq_max_horvath_probe}")
    logger.info(f"freq_max_buccino: {args.freq_max_buccino}")
    logger.info(f"save data path: {args.save_data_path}")
    logger.info(f"xlim: {args.xlim}")

    print("available cpus:", multiprocessing.cpu_count())
    print("available gpus:", torch.cuda.is_available())

    # recording configs: (name, raw_path, freq_max, is_buccino)
    configs = [
        ("m", args.recording_path_marques, args.freq_max_marques, False),
        ("ns", args.recording_path_npx_spont, args.freq_max_npx_spont, False),
        ("ne", args.recording_path_npx_evoked, args.freq_max_npx_evoked, False),
        ("d1", args.recording_path_dense_probe, args.freq_max_dense_probe, False),
        ("h1", args.recording_path_horvath_probe, args.freq_max_horvath_probe, False),
    ]
    if args.layer == "L5":
        configs.append(("b", args.recording_path_buccino, args.freq_max_buccino, True))

    # -------------------------------------------------------------------------
    # Pass 0: compute full-recording MAD per site for each recording
    # -------------------------------------------------------------------------
    logger.info("Pass 0: computing full-recording MAD per site ...")
    all_mads = {}

    for name, raw_path, freq_max, is_buccino in configs:
        logger.info(f"  MAD for {name} ...")
        Recording = load_and_preprocess(
            raw_path=raw_path,
            freq_min=args.freq_min,
            freq_max=freq_max,
            compress_to_int16=args.compress_to_int16,
            max_duration=args.max_duration,
            layer=args.layer,
            is_buccino=is_buccino,
        )
        all_mads[name] = compute_mad_by_site(Recording)
        del Recording
        gc.collect()
        logger.info(f"  Done MAD for {name}: shape {all_mads[name].shape}")

    # -------------------------------------------------------------------------
    # Pass 1: find global ANR range across all recordings
    # -------------------------------------------------------------------------
    logger.info("Pass 1: computing global ANR range ...")
    global_min, global_max = np.inf, -np.inf

    for name, raw_path, freq_max, is_buccino in configs:
        logger.info(f"  Range scan for {name} ...")
        Recording = load_and_preprocess(
            raw_path=raw_path,
            freq_min=args.freq_min,
            freq_max=freq_max,
            compress_to_int16=args.compress_to_int16,
            max_duration=args.max_duration,
            layer=args.layer,
            is_buccino=is_buccino,
        )
        mn, mx = get_anr_range(
            Recording,
            mads=all_mads[name],
            chunk_sites=CHUNK_SITES,
            chunk_samples=CHUNK_SAMPLES,
        )
        global_min = min(global_min, mn)
        global_max = max(global_max, mx)
        del Recording
        gc.collect()
        logger.info(f"  {name}: range [{mn:.3f}, {mx:.3f}]")

    # compute common bins
    steps = (global_max - global_min) / N_BINS
    bins = np.arange(global_min, global_max + steps / 2, steps)
    logger.info(f"Global ANR range: [{global_min:.3f}, {global_max:.3f}]")
    logger.info(f"Bins: {bins}")

    # -------------------------------------------------------------------------
    # Pass 2: compute histogram counts chunk by chunk
    # -------------------------------------------------------------------------
    logger.info("Pass 2: computing histogram counts ...")
    all_counts = {}

    for name, raw_path, freq_max, is_buccino in configs:
        logger.info(f"  Histogram for {name} ...")
        Recording = load_and_preprocess(
            raw_path=raw_path,
            freq_min=args.freq_min,
            freq_max=freq_max,
            compress_to_int16=args.compress_to_int16,
            max_duration=args.max_duration,
            layer=args.layer,
            is_buccino=is_buccino,
        )
        all_counts[name] = compute_histogram_counts_by_chunk(
            Recording,
            mads=all_mads[name],
            bins=bins,
            chunk_sites=CHUNK_SITES,
            chunk_samples=CHUNK_SAMPLES,
        )
        del Recording
        gc.collect()
        logger.info(f"  Done histogram for {name}")

    # -------------------------------------------------------------------------
    # Pass 3: compute PDF stats from histogram counts
    # -------------------------------------------------------------------------
    logger.info("Pass 3: computing PDF stats ...")
    means, cis = {}, {}
    for name, _, _, _ in configs:
        means[name], cis[name] = counts_to_pdf(all_counts[name])
    logger.info("Done computing PDF stats")

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------
    pm = {"linestyle": "-", "linewidth": 1, "marker": "None"}
    fig, ax = plt.subplots(1, 1, figsize=(1.9, 1.5))

    if args.layer != "L5":
        ax = amp.plot_snr_pdf_all(
            ax,
            means["m"],
            means["ns"],
            means["ne"],
            cis["m"],
            cis["ns"],
            cis["ne"],
            bins,
            COLOR_NV,
            COLOR_NS,
            COLOR_NE,
            pm,
            save_path=args.save_data_path + "_npx",
        )
        ax = amp.plot_snr_pdf_all(
            ax,
            means["h1"],
            means["d1"],
            [0],
            cis["h1"],
            cis["d1"],
            [0],
            bins,
            COLOR_HV,
            COLOR_HS,
            [0],
            pm,
            save_path=args.save_data_path + "_dense",
        )
    else:
        amp.plot_anr_pdf_l5(
            ax,
            means["m"],
            means["ns"],
            means["ne"],
            means["b"],
            cis["m"],
            cis["ns"],
            cis["ne"],
            cis["b"],
            bins,
            COLOR_NV,
            COLOR_NS,
            COLOR_NE,
            COLOR_NB,
            pm,
            save_path=args.save_data_path + "_npx",
        )
        amp.plot_anr_pdf_l5(
            ax,
            means["h1"],
            means["d1"],
            [0],
            [0],
            cis["h1"],
            cis["d1"],
            [0],
            [0],
            bins,
            COLOR_HV,
            COLOR_HS,
            [0],
            [0],
            pm,
            save_path=args.save_data_path + "_dense",
        )

    # format plot
    if args.xlim:
        xmin = int(args.xlim[0])
        xmax = int(args.xlim[1])
    else:
        xmin, xmax = ax.get_xlim()
    ax.set_xticks(
        [np.floor(xmin).astype(int), 0, np.ceil(xmax).astype(int)],
        [np.floor(xmin).astype(int), 0, np.ceil(xmax).astype(int)],
    )
    ax.set_xlim([np.floor(xmin), np.ceil(xmax)])

    labels = [f"$10^{{{int(np.log10(t))}}}$" for t in yticks]
    ax.set_yticks(yticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(ylims)
    fig.tight_layout(**tight_layout_cfg)
    plt.savefig(args.save_fig_path, **savefig_cfg)
    logger.info("Saved ANR plot.")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Compute ANR distributions")

    parser.add_argument(
        "--recording-path-marques", default="./dataset/00_raw/recording_marques_smith"
    )
    parser.add_argument(
        "--recording-path-npx-spont", default="./dataset/00_raw/recording_npx_spont"
    )
    parser.add_argument(
        "--recording-path-npx-evoked", default="./dataset/00_raw/recording_npx_evoked"
    )
    parser.add_argument(
        "--recording-path-dense-probe",
        default="",
    )
    parser.add_argument(
        "--recording-path-horvath-probe",
        default="",
    )
    parser.add_argument(
        "--recording-path-buccino", default="./dataset/00_raw/recording_buccino"
    )
    parser.add_argument("--max-duration", type=int, default=1800)
    parser.add_argument("--compress-to-int16", type=bool, default=False)
    parser.add_argument("--freq-min", type=int, default=None)
    parser.add_argument("--freq-max-marques", type=int, default=None)
    parser.add_argument("--freq-max-npx-spont", type=int, default=None)
    parser.add_argument("--freq-max-npx-evoked", type=int, default=None)
    parser.add_argument("--freq-max-dense-probe", type=int, default=None)
    parser.add_argument("--freq-max-horvath-probe", type=int, default=None)
    parser.add_argument("--freq-max-buccino", type=int, default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument(
        "--save-data-path", default="./dataset/01_intermediate/anrs/anrs.npz"
    )
    parser.add_argument(
        "--save-fig-path", default="./figures/08_source_data/fig2q/fig2q.svg"
    )
    parser.add_argument("--xlim", nargs="+", help="list of layers to analyse")

    args = parser.parse_args()

    # run and track peak RAM usage
    try:
        t0 = time.time()
        mem_usage = memory_usage((main, (args,)), max_usage=True)
        print(f"Peak RAM usage: {mem_usage / 1024:.2f} GB")
        print(f"All completed in: {np.round(time.time()-t0, 2)} secs")

    # else notify of error
    except Exception as e:
        error_msg = traceback.format_exc()
        notify(f"❌ Crash:\n{error_msg}")
        raise
