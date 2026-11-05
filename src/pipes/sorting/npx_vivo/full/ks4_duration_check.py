"""Kilosort 4 on a whole recording (default: Marques-Smith, 20.25 min; or the spontaneous simulation, 34.3 min), to check whether the median firing rate depends on the duration of the sorted recording

author: laquitainesteeve@gmail.com

Same sorter parameters as src/pipes/sorting/npx_vivo/10m/ks4.py (batch size 10,000, dminx 25.6 um), on dataset/00_raw/recording_marques_smith converted to int16.
Writes dataset/01_intermediate/sorting/marques/SortingKS4_full_check (the sorting, with the properties KSLabel, Amplitude, ContamPct, the peak channel
"contact" and its "layer") and deletes the temporary whitened binary of Kilosort afterwards. The 10-minute sorting SortingKS4_10m is not touched.

Usage (environment envs/spikebias, GPU; free disk space of at least 90 GB):

    cd /home/steeve/steeve/epfl/code/spikebias
    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/sorting/npx_vivo/full/ks4_duration_check.py [--run npx_spont]
"""

import os
import shutil
import time

import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
import spikeinterface.sorters as ss

import argparse

# name: (recording, output folder, free disk space needed in GB: an int16 copy of the recording and the whitened data of Kilosort)
RUNS = {
    "marques": ("dataset/00_raw/recording_marques_smith", "dataset/01_intermediate/sorting/marques", 90),
    "npx_spont": ("dataset/00_raw/recording_npx_spont", "dataset/01_intermediate/sorting/npx_spont", 135),
}

SORTER_PARAMS = {
    "batch_size": 10000, "nblocks": 1, "Th_universal": 9, "Th_learned": 8, "do_CAR": True, "invert_sign": False, "nt": 61, "artifact_threshold": None,
    "nskip": 25, "whitening_range": 32, "binning_depth": 5, "sig_interp": 20, "nt0min": None, "dmin": None, "dminx": 25.6, "min_template_size": 10,
    "template_sizes": 5, "nearest_chans": 10, "nearest_templates": 100, "templates_from_data": True, "n_templates": 6, "n_pcs": 6, "Th_single_ch": 6,
    "acg_threshold": 0.2, "ccg_threshold": 0.25, "cluster_downsampling": 20, "cluster_pcs": 64, "duplicate_spike_bins": 15, "do_correction": True,
    "keep_good_only": False, "save_extra_kwargs": False, "skip_kilosort_preprocessing": False, "scaleproc": None,
}


def peak_channels(output_folder, n_units_ids):
    """Channel of the largest template amplitude of each cluster (most frequent template of the cluster)"""
    spike_templates = np.load(f"{output_folder}/spike_templates.npy").ravel()
    spike_clusters = np.load(f"{output_folder}/spike_clusters.npy").ravel()
    templates = np.load(f"{output_folder}/templates.npy")                       # (n templates, nt, n channels)
    chan_map = np.load(f"{output_folder}/channel_map.npy").ravel()
    peak = np.abs(templates).max(axis=1).argmax(axis=1)                         # index into chan_map per template
    out = {}
    for u in n_units_ids:
        t = np.bincount(spike_templates[spike_clusters == u]).argmax()
        out[u] = int(chan_map[peak[t]])
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", choices=list(RUNS), default="marques")
    args = parser.parse_args()
    REC_PATH, OUT_DIR, MIN_FREE_GB = RUNS[args.run]
    KS_OUTPUT = f"{OUT_DIR}/ks4_full_check_output"
    SORTING_PATH = f"{OUT_DIR}/SortingKS4_full_check"
    t0 = time.time()
    free = shutil.disk_usage(OUT_DIR).free / 1e9
    assert free > MIN_FREE_GB, f"only {free:.0f} GB free"
    rec = si.load_extractor(REC_PATH)
    print("recording", rec, flush=True)
    rec_int16 = spre.astype(rec, "int16")
    sorting = ss.run_sorter(sorter_name="kilosort4", recording=rec_int16, remove_existing_folder=True, output_folder=KS_OUTPUT, verbose=True, **SORTER_PARAMS)
    sorting = sorting.remove_empty_units()
    print(f"sorted {len(sorting.unit_ids)} units in {time.time() - t0:.0f} s", flush=True)
    out_folder = f"{KS_OUTPUT}/sorter_output"
    peaks = peak_channels(out_folder, sorting.unit_ids)
    contact = np.array([peaks[u] for u in sorting.unit_ids])
    layers = np.array(rec.get_property("layers"))[contact]
    sorting.set_property("contact", contact)
    sorting.set_property("layer", layers)
    shutil.rmtree(SORTING_PATH, ignore_errors=True)
    sorting = sorting.save(folder=SORTING_PATH)
    print("properties:", sorting.get_property_keys(), flush=True)
    for f in ("temp_wh.dat", "recording.dat"):
        p = f"{out_folder}/{f}"
        if os.path.exists(p):
            os.remove(p)
            print("deleted", p, flush=True)
    print(f"done in {time.time() - t0:.0f} s", flush=True)
