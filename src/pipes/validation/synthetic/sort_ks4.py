"""Sort the first 10 minutes of a synthetic simulation (S2-S5) with Kilosort 4.0 (Supplementary Fig. 6d-e)

author: laquitainesteeve@gmail.com

Same sorting parameters as src/pipes/sorting/npx_synth3/ks4.py (batch size 10,000, dminx 25.6 um). The sorting is written in
dataset/01_intermediate/sorting/synth<N>/SortingKS4_10m with the metadata used by the analyses (layer, firing rates, KSLabel). The
waveform extraction of the paper's script is skipped: it is not needed for the yield and the sorting accuracy.

Usage (environment envs/spikebias, GPU):

    cd /home/steeve/steeve/epfl/code/spikebias
    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/validation/synthetic/sort_ks4.py --name s3
"""

import argparse
import os
import shutil
import sys

import numpy as np
import spikeinterface as si
import torch

PROJ_PATH = "/home/steeve/steeve/epfl/code/spikebias/"
os.chdir(PROJ_PATH)
sys.path.append(os.path.join(PROJ_PATH, "src"))
from src.nodes.sorting import sort_and_postprocess_10m
from src.nodes.validation import firing_rate

REC_SECS = 600
SORTER = "kilosort4"
NUMBERS = {"s2": 2, "s3": 3, "s4": 4, "s5": 5}
LAYER = "L5"  # all sites are in layer 5

# these are the default parameters for spikeinterface 0.100.5 (as in src/pipes/sorting/npx_synth3/ks4.py)
# - we set batch_size to 10,000 instead of 60,0000 due to memory constrains
# - we set dminx to 25.6 um instead of None
SORTER_PARAMS = {
    "batch_size": 10000,
    "nblocks": 1,
    "Th_universal": 9,
    "Th_learned": 8,
    "do_CAR": True,
    "invert_sign": False,
    "nt": 61,
    "artifact_threshold": None,
    "nskip": 25,
    "whitening_range": 32,
    "binning_depth": 5,
    "sig_interp": 20,
    "nt0min": None,
    "dmin": None,
    "dminx": 25.6,
    "min_template_size": 10,
    "template_sizes": 5,
    "nearest_chans": 10,
    "nearest_templates": 100,
    "templates_from_data": True,
    "n_templates": 6,
    "n_pcs": 6,
    "Th_single_ch": 6,
    "acg_threshold": 0.2,
    "ccg_threshold": 0.25,
    "cluster_downsampling": 20,
    "cluster_pcs": 64,
    "duplicate_spike_bins": 15,
    "do_correction": True,
    "keep_good_only": False,
    "save_extra_kwargs": False,
    "skip_kilosort_preprocessing": False,
    "scaleproc": None,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=list(NUMBERS), help="simulation")
    parser.add_argument("--keep-temp", action="store_true", help="keep the int16 binary copy of the recording and Kilosort's output")
    args = parser.parse_args()
    n = NUMBERS[args.name]
    rec_path = f"dataset/00_raw/recording_synth{n}"
    sorting_path = f"dataset/01_intermediate/sorting/synth{n}/SortingKS4_10m"
    temp = f"./temp/pros_of_details/npx_synth{n}"
    cfg = {
        "probe_wiring": {"full": {"output": rec_path}},
        "preprocessing": {"full": {"output": {"trace_file_path": rec_path}}},
        "sorting": {"sorters": {SORTER: {"10m": {"output": sorting_path, "sort_output": f"{temp}/KS4_output_10m_RTX5090/"}}}},
        "postprocessing": {"waveform": {"sorted": {"study": {SORTER: {"10m": f"{temp}/study_ks4_10m_RTX5090/"}}}}},
    }
    os.makedirs(os.path.dirname(sorting_path), exist_ok=True)
    torch.cuda.empty_cache()

    # spike sort the first 10 minutes (the int16 binary copy of the recording is written in the temporary folder)
    sort_and_postprocess_10m(cfg, SORTER, SORTER_PARAMS, duration_sec=REC_SECS, is_sort=True, is_postpro=False, extract_wvf=False,
                             copy_binary_recording=True, remove_bad_channels=True, bad_channel_ids=None)

    # metadata of the sorted units
    Sorting = si.load_extractor(sorting_path)
    Sorting.set_property("firing_rates", np.array([len(Sorting.get_unit_spike_train(u)) / REC_SECS for u in Sorting.unit_ids]))
    Sorting.set_property("layer", np.array([LAYER] * len(Sorting.unit_ids)))
    shutil.rmtree(sorting_path, ignore_errors=True)
    Sorting.save(folder=sorting_path)
    print("Total units:", len(Sorting.unit_ids), "| single units:", int(sum(Sorting.get_property("KSLabel") == "good")))

    # free the disk
    if not args.keep_temp:
        shutil.rmtree(temp, ignore_errors=True)
