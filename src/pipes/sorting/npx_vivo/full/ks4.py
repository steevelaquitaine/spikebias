"""sort and postprocess 20 min of marques-smith with Kilosort 4.0

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 07.05.2024

usage:

    sbatch cluster/sorting/npx_vivo/ks4.sh

Note:
    - We do no preprocessing as Kilosort4 already preprocess the traces with
    (see code preprocessDataSub()):
    - we set minFR and minfr_goodchannels to 0
    
duration: 
    sorting (21 mins) + postprocessing (16 mins)

% 1) conversion to int16;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: 
- to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints
- crashes with "Maximum variable size allowed on the device is exceeded." with "minFR"
and "minfr_goodchannels" set to 0.

"""
import logging
import logging.config
import shutil
from time import time
import spikeinterface as si
import spikeinterface.sorters as ss
import spikeinterface.preprocessing as spre
import yaml
import pandas as pd

from src.nodes.utils import get_config
from src.nodes.validation import firing_rate
from src.nodes.postpro import get_waveforms
from src.nodes.postpro import waveform
import spikeinterface.core.template_tools as ttools
from spikeinterface import WaveformExtractor, extract_waveforms

# SAVE recording as int16
SAVE_REC_INT16 = True   # once; exists

# SET PARAMETERS
# these are the default parameters
# for the version of spikeinterface used
sorter_params = {
    "batch_size": 30000, #60000,
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
    "dminx": 25.6, #None,
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

# pipeline
SORT = True
EXTRACT_WAVEFORM = True

# waveform params
MS_BEFORE = 3
MS_AFTER = 3

# SETUP CONFIG
exp = "vivo_marques"
run = "c26"
sorter = "kilosort4"

data_conf, _ = get_config(exp, run).values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ PATHS
WIRED_PATH = data_conf["probe_wiring"]["output"]
PREP_PATH = data_conf["preprocessing"]["output"]["trace_file_path"]

# KILOSORT WRITE PATHS
SORTING_PATH = data_conf["sorting"]["sorters"][sorter]["output"]
OUTPUT_PATH = data_conf["sorting"]["sorters"][sorter]["ks4_output"]

# postprocessing
STUDY_PATH = data_conf["postprocessing"]["waveform"]["sorted"]["study"][sorter]

# n_jobs =-1 uses ProcessPoolExecutor on all cores (typically 72 on our machines)
# takes 5 mins (all cores: -1) instead of 30 mins (2 jobs)
job_kwargs = dict(n_jobs=-1, chunk_duration="1s", progress_bar=True)


def sort():
    """spike sorting
    """
    
    t0 = time()
    
    Recording = si.load_extractor(WIRED_PATH)

    # convert as int16
    Recording = spre.astype(Recording, "int16")  # convert to int16 for KS4
    logger.info("Done converting as int16 in: %s", round(time() - t0, 1))

    # run sorting (default parameters)
    t0 = time()

    sorting_KS4 = ss.run_sorter(sorter_name=sorter,
                                recording=Recording,
                                remove_existing_folder=True,
                                output_folder=OUTPUT_PATH,
                                verbose=True,
                                **sorter_params)

    # remove empty units (Samuel Garcia's advice)
    sorting_KS4 = sorting_KS4.remove_empty_units()
    logger.info("Done running kilosort4 in: %s", round(time() - t0, 1))

    # write
    shutil.rmtree(SORTING_PATH, ignore_errors=True)
    sorting_KS4.save(folder=SORTING_PATH)
    logger.info("Done saving kilosort4 in: %s", round(time() - t0, 1))


def save_sorted_units_site_and_layers(
        Sorting,
        wired_path: str,
        prep_path: str,
        study_path: str,
        sorting_path: str,
        extract_wfs: bool,
        save: bool
        ):
    """save sorted units site and layers metadata to Sorting Extractor

    Args:
        Sorting: Sorting extractor
        exp: experiment: "vivo_marques"
        run: run e.g., c26
        wired_path (str): probe-wired raw recording path
        prep_path (str): preprocessed recording path
        study_path (str): _description_
        sorting_path (str): _description_
        extract_wfs (bool): _description_
        save (bool): _description_

    Returns:
        _type_: _description_
    """
    # load recording
    Recording = si.load_extractor(prep_path)
    
    # extract waveforms
    # from preprocessed trace
    if extract_wfs:
        
        # extract waveforms
        #get_waveforms.run(exp, run, sorter, prep_path)
        #we = waveform.load(recording, study_path, MS_BEFORE, MS_AFTER)
        we = extract_waveforms(
            Recording,
            Sorting,
            study_path,
            sparse=False,
            ms_before=3.,
            ms_after=3.,
            max_spikes_per_unit=500,
            overwrite=True,
            **job_kwargs
        )        
        
        # get nearest sites (max amplitude spike)
        max_chids = ttools.get_template_extremum_channel(we, peak_sign="both")

    # map units to sites
    unit_sites = pd.DataFrame.from_dict(max_chids, orient="index")
    unit_sites.columns = ["site"]

    # get site layers 
    Recording = si.load_extractor(wired_path)
    site_layers = pd.DataFrame(
        Recording.get_property("layers"), index=Recording.channel_ids, columns=["layer"]
    )

    # map sorted units to sites and layers
    # site_and_layer (index: site, columns: layer and unit)
    unit_sites = pd.DataFrame.from_dict(max_chids, orient="index")
    unit_sites.columns = ["site"]
    unit_sites["unit"] = unit_sites.index
    unit_sites.index = unit_sites["site"].values
    unit_sites = unit_sites.drop(columns=["site"])
    site_and_layer = pd.merge(site_layers, unit_sites, left_index=True, right_index=True)
    site_and_layer = site_and_layer.sort_values(by="unit")

    # standardize layer names
    site_and_layer["layer"] = site_and_layer["layer"].replace(
        "Outsideofthecortex", "Outside"
    )

    # Write metadata to Sorting Extractor
    # - Sanity check
    # - Save layer as a property
    # - Save nearest site as a property
    assert all(Sorting.unit_ids == site_and_layer["unit"]), "units should match"
    Sorting.set_property("layer", site_and_layer["layer"].tolist())
    Sorting.set_property("contact", site_and_layer.index.tolist())
    
    # save
    if save:
        shutil.rmtree(sorting_path, ignore_errors=True)
        Sorting.save(folder=sorting_path)
    return Sorting


def postprocess():
    
    # load sorting extractor
    Sorting = si.load_extractor(SORTING_PATH)

    # save sorted unit firing rates
    Sorting = firing_rate.label_firing_rates(Sorting, PREP_PATH, SORTING_PATH, save=False)

    # save sorted unit layers and sites
    # - extract spike waveform
    # - get nearest site and layer
    # - save its layer
    Sorting = save_sorted_units_site_and_layers(Sorting, WIRED_PATH, PREP_PATH, STUDY_PATH, SORTING_PATH, extract_wfs=EXTRACT_WAVEFORM, save=True)


# get Spikeinterface Recording object
t0 = time()

# sort
if SORT:
    sort()
    logger.info(f"Done sorting with {sorter} in: %s", round(time() - t0, 1))

# postprocess
postprocess()
logger.info(f"Done postprocessing for {sorter}- metadata written in: %s", round(time() - t0, 1))