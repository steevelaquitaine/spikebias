"""Simulate one of the intermediate synthetic recordings S2-S5 that link Buccino's model to our biophysical model with MEArec
(Supplementary Fig. 6c-e)

author: laquitainesteeve@gmail.com

The simulation parameters are those of the paper (assets/mearec/<simulation>/input_params.json); the cell templates (19,500 templates
for 13 cell models) are in temp/pros_of_details/templates.h5. The convolution and the noise are computed by chunks of 10 seconds whose
random seeds are drawn in advance, so the traces do not depend on the number of parallel jobs.

MEArec version: the parameters of S2-S5 (no `filter_mode`) come from MEArec 1.8.0, not from 1.9.0 (environment envs/mearec_si_0_103_0).
In 1.8.0 the band-pass filter step ADDS the filtered traces to the unfiltered traces (recordings += filtered chunk), in 1.9.0 it replaces
them. The recordings of the paper (dandiset 001250, sub-mearec-synthetics-s2) are therefore unfiltered traces plus their band-pass filtered
version: their spikes are 1.73 times larger (1 + 0.73) than the filtered ones and their white noise (10 uV) is unchanged above 6 kHz.
MEArec 1.8.0 is used (temp/mearec_1_8_0, installed and patched by run_all.py), and its filter step is
run sequentially after the (parallel) convolution and noise steps. Set MEAREC_SKIP_SPIKE_TRACES=1 to skip the 106 GB of spike traces (patch).

Usage (environment envs/mearec_si_0_103_0 with MEArec 1.8.0 first on the path):

    cd /home/steeve/steeve/epfl/code/spikebias
    MEAREC_SKIP_SPIKE_TRACES=1 PYTHONPATH=$PWD:$PWD/temp/mearec_1_8_0 envs/mearec_si_0_103_0/bin/python src/pipes/validation/synthetic/simulate_mearec.py --name s3 --n-jobs 8

Outputs: dataset/00_raw/recording_synth<N>/ (traces, float32) and dataset/00_raw/ground_truth_synth<N>/ (ground truth SortingExtractor)
"""

import argparse
import gc
import json
import os
import shutil
import time

import numpy as np

from src.nodes.validation import synthetic_simulation as ss

# MEArec 1.8.0 uses numpy aliases that were removed from numpy 1.24
for alias, builtin in (("float", float), ("int", int), ("bool", bool), ("object", object)):
    if not hasattr(np, alias):
        setattr(np, alias, builtin)

# simulation: (folder of the parameters in assets/mearec/, number in the output folders)
SIMULATIONS = {"s2": ("synth2", 2), "s3": ("synth3", 3), "s4": ("buccino_s4", 4), "s5": ("synth5", 5)}
TEMPLATES_PATH = "temp/pros_of_details/templates.h5"
TMP_FOLDER = "temp/pros_of_details/tmp_mearec_{name}"   # one temporary folder per simulation (simulations can run at the same time)
LAYER = "L5"  # all sites are in layer 5, as in Buccino's model


def main(name: str, n_jobs: int, test: bool = False):
    import MEArec as mr
    import spikeinterface as si
    import spikeinterface.extractors as se

    t0 = time.time()
    tmp_folder = TMP_FOLDER.format(name=name)
    folder, number = SIMULATIONS[name]
    number = f"{number}_test" if test else number
    rec_path = f"dataset/00_raw/recording_synth{number}"
    gt_path = f"dataset/00_raw/ground_truth_synth{number}"
    h5_path = f"temp/pros_of_details/recording_synth{number}.h5"

    # simulation parameters of the paper
    with open(f"assets/mearec/{folder}/input_params.json") as f:
        rec_prms = json.load(f)

    # the filter step of MEArec 1.8.0 is applied below (after the parallel steps), not by gen_recordings
    filter_params = (np.array(rec_prms["recordings"]["filter_cutoff"], dtype=float), rec_prms["recordings"]["filter_order"], rec_prms["recordings"]["chunk_duration"])
    rec_prms["recordings"]["filter"] = False
    if test:        # small simulation to test the pipeline
        rec_prms["spiketrains"].update({"n_exc": 12, "n_inh": 3, "duration": 20})

    # simulate (convolution and noise by chunks, in parallel)
    os.makedirs(tmp_folder, exist_ok=True)
    rec_gen = mr.gen_recordings(params=rec_prms, templates=TEMPLATES_PATH, n_jobs=n_jobs, tmp_mode="memmap", tmp_folder=tmp_folder)
    print(f"{name} simulated in {time.time() - t0:.0f} s", flush=True)

    # filter step of MEArec 1.8.0: the band-pass filtered traces of each chunk are ADDED to the traces (chunks in order, so that the
    # padding of a chunk contains the modified end of the previous chunk, as in MEArec 1.8.0)
    assert mr.__version__ == "1.8.0", "use MEArec 1.8.0 (see the docstring)"
    import quantities as pq
    from MEArec.generators.recgensteps import chunk_apply_filter_
    from MEArec.generators.recordinggenerator import make_chunk_indexes

    cutoff, order, chunk_duration = filter_params
    fs = rec_gen.info["recordings"]["fs"] * pq.Hz
    cutoff = cutoff * pq.Hz
    recordings = rec_gen.recordings
    pad_samples = 3 * int((1.0 / cutoff[0] * fs).magnitude)
    for ch, (i_start, i_stop) in enumerate(make_chunk_indexes(rec_gen.info["recordings"]["duration"] * pq.s, chunk_duration * pq.s, fs)):
        out = chunk_apply_filter_(ch, i_start, i_stop, fs.rescale("Hz").magnitude, None, recordings, pad_samples, cutoff, order, recordings.dtype)
        recordings[i_start:i_stop] += out["filtered_chunk"].astype(recordings.dtype)
    recordings.flush()
    print(f"{name} filter step done in {time.time() - t0:.0f} s", flush=True)

    # check the simulation parameters against those saved by the paper's notebook
    with open(f"assets/mearec/{folder}/output_params.json") as f:
        out_prms = json.load(f)
    differences = [k for k in out_prms if json.dumps(out_prms[k], sort_keys=True) != json.dumps(json.loads(json.dumps(rec_gen.params.get(k), default=str)), sort_keys=True)]
    print("parameters differing from assets/mearec/%s/output_params.json (expected: recordings.filter, applied above):" % folder, differences, flush=True)

    # free the disk: MEArec stores the voltage trace of every neuron (spike_traces, 106 GB for 1,388 neurons), which we do not use
    spike_traces_files = [p for p in rec_gen._to_remove_on_delete if "spike_traces" in str(p)]
    rec_gen.spike_traces = np.array([])
    gc.collect()
    for p in spike_traces_files:
        if os.path.exists(p):
            os.remove(p)

    # save the simulated recording and load its extractors
    mr.save_recording_generator(rec_gen, h5_path)
    Recording = se.MEArecRecordingExtractor(h5_path)
    Recording.set_property("layers", np.array([LAYER] * Recording.get_num_channels()))
    Gt = se.MEArecSortingExtractor(h5_path)
    Recording, Gt = ss.set_metadata_to_mearec_simulations(rec_gen, Recording, Gt)
    Recording = ss.standardize_site_location_property(Recording)
    Gt = ss.standardize_metadata(Gt)

    # delete the temporary files
    del rec_gen
    gc.collect()
    shutil.rmtree(tmp_folder, ignore_errors=True)

    # save the extractors (float32 traces)
    Recording.save(folder=rec_path, n_jobs=min(n_jobs, 16), verbose=True, progress_bar=False, overwrite=True, dtype="float32", chunk_size=50000)
    Gt.save(folder=gt_path, overwrite=True)
    os.remove(h5_path)
    print(f"{name} saved in {time.time() - t0:.0f} s:", len(si.load_extractor(gt_path).unit_ids), "units", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True, choices=list(SIMULATIONS), help="simulation")
    parser.add_argument("--n-jobs", type=int, default=8, help="number of parallel jobs of the simulation")
    parser.add_argument("--test", action="store_true", help="simulate 15 neurons for 20 seconds (pipeline test, outputs *_test)")
    args = parser.parse_args()
    main(args.name, args.n_jobs, args.test)
