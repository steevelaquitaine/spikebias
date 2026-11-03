"""Compute the features of the ground truth neurons and of their waveforms for a synthetic simulation
(Buccino, MEArec S2-S5) or for the biophysical simulation of the spontaneous regime (Supplementary Fig. 6c)

author: laquitainesteeve@gmail.com

Usage (environment envs/mearec_si_0_103_0, as in the paper's notebook):

    cd /home/steeve/steeve/epfl/code/spikebias
    PYTHONPATH=$PWD envs/mearec_si_0_103_0/bin/python src/pipes/validation/synthetic/compute_features.py --name buccino

The features are saved in dataset/01_intermediate/supplementary/synthetic_features/<name>.npz
"""

import argparse
import os
import time
import warnings

import numpy as np
import spikeinterface as si
import spikeinterface.preprocessing as spre
from spikeinterface.core.sparsity import compute_sparsity
from spikeinterface.curation import remove_excess_spikes

from src.nodes.validation import synthetic_features as sf
from src.nodes.validation import synthetic_simulation as ss

warnings.filterwarnings("ignore")

DURATION_S = 600  # duration of the recording period considered (seconds)
OUT_FOLDER = "dataset/01_intermediate/supplementary/synthetic_features"

# recording and ground truth of each simulation
PATHS = {
    "ns": ("dataset/00_raw/recording_npx_spont", "dataset/00_raw/ground_truth_npx_spont"),
    "buccino": ("dataset/00_raw/recording_buccino", "dataset/00_raw/sorting_buccino"),
    "s2": ("dataset/00_raw/recording_synth2", "dataset/00_raw/ground_truth_synth2"),
    "s3": ("dataset/00_raw/recording_synth3", "dataset/00_raw/ground_truth_synth3"),
    "s4": ("dataset/00_raw/recording_synth4", "dataset/00_raw/ground_truth_synth4"),
    "s5": ("dataset/00_raw/recording_synth5", "dataset/00_raw/ground_truth_synth5"),
}

# sorting analyzer extensions
EXTENSION_PARAMS = {"random_spikes": {"seed": 0}, "waveforms": {"ms_before": 3.0, "ms_after": 2.0}, "spike_amplitudes": {"peak_sign": "neg"}}


def load(name: str):
    """load the recording and the ground truth of a simulation, as in the paper's notebook"""
    rec_path, gt_path = PATHS[name]
    Recording = si.load_extractor(rec_path)
    Gt = si.load_extractor(gt_path)

    if name == "ns":
        # preprocessing of the biophysical simulation: int16, 300 Hz high-pass, median common reference (first 10 minutes)
        Recording = ss.standardize_biophy_site_location_property(Recording)  # 3D site locations
        Recording = spre.scale(Recording, gain=1.0, offset=0.0)       # in-memory copy of the traces (astype rounds the traces in place, which fails on read-only memory maps)
        Recording = spre.astype(Recording, "int16")
        Recording = spre.bandpass_filter(Recording, freq_min=300, freq_max=19999)
        Recording = spre.common_reference(Recording, reference="global", operator="median")
        Recording = Recording.frame_slice(start_frame=0, end_frame=int(Recording.get_sampling_frequency() * DURATION_S))
        Recording.set_property("location", Recording.get_property("location")[:, :2])
        Gt = remove_excess_spikes(Gt, Recording)
        Gt = Gt.frame_slice(start_frame=0, end_frame=int(Recording.get_sampling_frequency() * DURATION_S))
    else:
        if name == "buccino":
            # original Buccino simulation (already band-pass filtered 300-6000 Hz): 3D site locations (x=0), cell locations to x, y, z
            Recording.set_property("location3D", np.insert(Recording.get_channel_locations(), 0, 0, axis=1))
            Gt = ss.standardize_metadata(Gt)
        Recording.set_property("location", Recording.get_property("location")[:, :2])  # 2D channel location for spatial metrics
    return Recording, Gt


def main(name: str, n_jobs: int):
    t0 = time.time()
    si.set_global_job_kwargs(n_jobs=n_jobs, chunk_duration="1s", progress_bar=False)

    # load
    Recording, Gt = load(name)
    print(name, "recording:", Recording.get_num_channels(), "sites,", Recording.get_total_duration(), "s | ground truth:", len(Gt.unit_ids), "units", flush=True)

    # sorting analyzer (waveforms, templates, noise levels, spike amplitudes)
    analyzer = si.create_sorting_analyzer(sorting=Gt, recording=Recording, format="memory")
    analyzer.compute(["random_spikes", "waveforms", "templates", "noise_levels", "spike_amplitudes"], extension_params=EXTENSION_PARAMS, save=False)
    analyzer.sparsity = compute_sparsity(templates_or_sorting_analyzer=analyzer, method="radius", radius_um=100)
    print("analyzer done in %.0f s" % (time.time() - t0), flush=True)

    # features
    spatial_metrics = sf.get_spatial_metrics(analyzer)
    features = {
        "cell_to_site_distance": sf.get_cell_to_site_distance(Recording, Gt),
        "cell_distance": sf.get_cell_to_cell_distances(Gt),
        "firing_rates": np.array(sf.get_firing_rates(Gt, duration_s=DURATION_S)),
        "firing_ranges": sf.get_firing_ranges(analyzer),
        "snrs": sf.get_waveform_snr(analyzer),
        "amplitudes": sf.get_waveform_amplitude_medians(analyzer),
        "mad_ratios": sf.get_mad_ratios(analyzer),
        "spreads_simi_weighted_distance": sf.get_spatial_spread_all_units(Recording, analyzer, peak_sign="neg"),
        "si_spread": np.array(spatial_metrics["spread"]),
        "repolarization_slope": np.array(spatial_metrics["repolarization_slope"]),
        "exp_decay": np.array(spatial_metrics["exp_decay"]),
        "peak_to_valley": np.array(spatial_metrics["peak_to_valley"]),
        "velocity_above": np.array(spatial_metrics["velocity_above"]),
        "velocity_below": np.array(spatial_metrics["velocity_below"]),
    }
    os.makedirs(OUT_FOLDER, exist_ok=True)
    np.savez(os.path.join(OUT_FOLDER, f"{name}.npz"), **features)
    print({k: v.shape for k, v in features.items()})
    print(f"{name} done in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=list(PATHS), help="simulation")
    parser.add_argument("--n-jobs", type=int, default=16, help="number of parallel jobs")
    args = parser.parse_args()
    main(args.name, args.n_jobs)
