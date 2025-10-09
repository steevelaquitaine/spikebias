"""set SortingExtractor metadata

author: steeve.laquitaine@epfl.ch

Returns:
    Sorting (SortingExtractor): _description_
    
Note:
- extracts waveforms in a WaveformExtractor:
  - open files limits: 4,096 in production partition (16,384 in interactive).
  - the hard (theoretical) limit is 131,072 on nodes on the prod partition
  - we max out limit as a limit of 16,384 is needed to open waveform files for 4,600 units
"""
import os 
import logging
import logging.config
from spikeinterface import extract_waveforms
import spikeinterface as si
import pandas as pd
import spikeinterface.core.template_tools as ttools
import shutil 
import yaml
import resource
from time import time
import socket
import numpy as np
from src.nodes.validation import firing_rate
from src.nodes import utils
from src.nodes.postpro import (
    cell_type,
)

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def set_sorting_duration(Sorting, duration):
    """set sorting duration metadata
    to retrieve use Sorting.annotate(key="sorting_duration")

    Args:
        Sorting (_type_): _description_
    
    Returns:
        _type_: _description_
    """
    Sorting.annotate(sorting_duration=str(duration))
    return Sorting


def set_hostname(Sorting):
    """set hostname metadata
    to retrieve use sorting.get_annotation(key="hostname")

    Args:
        Sorting (_type_): _description_

    Returns:
        _type_: _description_
    """
    Sorting.annotate(hostname=socket.gethostname())
    return Sorting


def set_site_and_layer(
        Sorting,
        Wired,
        prep_path: str,
        study_path: str,
        sorting_path: str,
        extract_wfs: bool,
        save: bool,
        job_kwargs: dict,
        ):
    """save sorted units site and layers metadata to Sorting Extractor
    - extract 500 action potential waveforms per unit
    - map each unit to the nearest site
    - map each unit to a layer
    - works on all recordings - does not require ground truth

    Args:
        Sorting: Sorting extractor
        Wired (RecordingExtractor):  Probe-wired recording with metadata properties:
        - "layers" (Wired.get_property("layers"))
        exp: experiment: "vivo_marques"
        run: run e.g., c26
        wired_path (str): probe-wired raw recording path
        prep_path (str): preprocessed recording path
        study_path (str): _description_
        sorting_path (str): _description_
        extract_wfs (bool): _description_
        save (bool): _description_

    Returns:
        Sorting (SortingExtractor): SortingExtractor updated with sorted unit metadata:
        - "contact": units' nearest contact (with extremum spike amplitude)
        - "layer": units' layer inferred from nearest contact
    """
    # load recording
    Preprocessed = si.load_extractor(prep_path)
    
    # extract waveforms
    # from preprocessed trace
    if extract_wfs:
        
        # report
        t0 = time()
        logger.info("Extracting waveforms into a WaveformExtractor...")
        
        # check system's opened file limit
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        logger.info("Opened file limits:")
        logger.info(f"- soft: {soft}")
        logger.info(f"- hard (max): {hard}")
        
        # max out limit for extract_waveforms
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
        logger.info(f"Max out limit")
        
        # extract waveforms
        # to load use: we = si.WaveformExtractor.load_from_folder(STUDY_ns)
        # dense waveforms are extracted first using a small number of spikes
        # remove existing study folder
        shutil.rmtree(study_path, ignore_errors=True)
        
        We = extract_waveforms(
            Preprocessed,
            Sorting,
            study_path,
            sparse=True, # False
            ms_before=3.0,
            ms_after=3.0,
            max_spikes_per_unit=500,
            unit_batch_size=200,
            overwrite=True,
            seed=0,
            **job_kwargs
        )
        logger.info(f"Done extracting waveforms in %s", round(time() - t0, 1))
    else:
        # load existing WaveformExtractor
        logger.info("Loading pre-existing WaveformExtractor")
        We = si.WaveformExtractor.load_from_folder(study_path)
        
        # get nearest sites (where the spike has its max amplitude)
    max_chids = ttools.get_template_extremum_channel(We, peak_sign="both")

    # map recording sites to sorted units
    t0 = time()
    logger.info("Saving layers and site contact metadata to SortingExtractor...")
    
    # - get site layers
    unit_sites = pd.DataFrame.from_dict(max_chids, orient="index")
    unit_sites.columns = ["site"]
    site_layers = pd.DataFrame(
        Wired.get_property("layers"), index=Wired.channel_ids, columns=["layer"]
    )

    # map sorted units to sites and layers
    # site_and_layer (index: site, columns: layer and unit)
    unit_sites = pd.DataFrame.from_dict(max_chids, orient="index", columns=["site"])
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
    assert len(Sorting.unit_ids) == len(site_and_layer["unit"]), "number of units should match."
    assert all(Sorting.unit_ids == site_and_layer["unit"]), "sorted unit ids should match."
    Sorting.set_property("layer", site_and_layer["layer"].tolist())
    Sorting.set_property("contact", site_and_layer.index.tolist())
    
    # save
    if save:
        shutil.rmtree(sorting_path, ignore_errors=True)
        Sorting.save(folder=sorting_path)
    logger.info("Saving done in %s", round(time() - t0, 1))
    return Sorting
def set_firing_rates(Sorting, duration: float, sorting_path: str, save: bool):
    """Save sorted units' firing rate property in Sorting Extractor

    Args:
        Sorting (Sorting Extractor):
        duration (float): recording duration in seconds
        save (bool): write or not

    Returns:
        Sorting Extractor : Saves Sorting Extractor with "firing_rates" property as metadata:
    """
    rates = []
    for unit_id in Sorting.get_unit_ids():
        st = Sorting.get_unit_spike_train(unit_id=unit_id)
        rates += [len(st) / duration]

    # add firing rates to properties
    Sorting.set_property("firing_rates", list(rates))

    # save
    if save:
        shutil.rmtree(sorting_path, ignore_errors=True)
        Sorting.save(folder=sorting_path)
    return Sorting


def postprocess(Recording, prep_path, sorting_path, 
                study_path, extract_wvf=True, 
                job_kwargs: dict={"n_jobs":-1}):
    """save metadata to SortingExtractor
    
    Args:
        Recording (): 
        prep_path (str): _description_
        sorting_path (str): _description_
        - SortingExtractor
        study_path (_type_): _description_
        extract_wvf (bool): _description_
        job_kwargs (dict): _description_    
    
    Returns:
    Updates and saves SortingExtractor with:
    - firing rate
    """
    # load sorting extractor
    Sorting = si.load_extractor(sorting_path)

    # save sorted unit firing rates
    Sorting = firing_rate.label_firing_rates(Sorting, Recording, sorting_path, False)

    # save sorted unit layers and sites
    # - extract spike waveform
    # - get nearest site and layer
    # - save its layer
    Sorting = set_site_and_layer(
        Sorting, Recording, prep_path, study_path,
        sorting_path, extract_wvf, True, job_kwargs
    )


def set_gt_metadata(gt_read_path: str, save_path: str, blueconfig_path, duration, save: bool):
    """Save metadata in Ground truth SortingExtractor
    - Blue Brain cell properties
    - firing rates

    Args:
        gt_read_path (str): _description_
        save_path (str): _description_
        blueconfig_path (_type_): _description_
        duration (_type_): _description_
        save (bool): _description_

    Returns:
        SortingExtractor: _description_
    """

    # Get ground truth SortingExtractor
    # - get first 10 minutes
    SortingTrue = si.load_extractor(gt_read_path)
    SortingTrue = SortingTrue.frame_slice(
        start_frame=0, end_frame=duration * SortingTrue.sampling_frequency
    )

    # set unit features metadata (includes layer)
    SortingTrue = cell_type.label_true_cell_properties(
        SortingTrue, blueconfig_path, gt_read_path, save=False
    )

    # set unit firing rate metadata
    SortingTrue = set_firing_rates(SortingTrue, duration, gt_read_path, save=False)

    # save
    if save:
        shutil.rmtree(save_path, ignore_errors=True)
        SortingTrue.save(folder=save_path)
    return SortingTrue


def get_gt_unit_meta_df(sorted_path: str):
    """get ground truth unit metadata

    Returns:
        pd.DataFrame: _description_
    """
    # load ground truth extractor
    Sorting = si.load_extractor(sorted_path)

    # record
    unit_id_all = Sorting.unit_ids.tolist()
    firing_rate_all = (
        Sorting.get_property("firing_rates").astype(np.float32).tolist()
    )
    layer_all = Sorting.get_property("layer").tolist()
    layer_all = utils.standardize_gt_layers(layer_all)

    # store in dataframe
    return pd.DataFrame(
        np.array(
            [
                layer_all,
                firing_rate_all,
            ]
        ).T,
        index=unit_id_all,
        columns=[
            "layer",
            "firing_rate",
        ],
    )

def get_gt_unit_meta_df_from_extractor(Sorting):
    """get ground truth unit metadata

    Returns:
        pd.DataFrame: _description_
    """
    # record
    unit_id_all = Sorting.unit_ids.tolist()
    firing_rate_all = (
        Sorting.get_property("firing_rates").astype(np.float32).tolist()
    )
    layer_all = Sorting.get_property("layer").tolist()
    layer_all = utils.standardize_gt_layers(layer_all)

    # store in dataframe
    return pd.DataFrame(
        np.array(
            [
                layer_all,
                firing_rate_all,
            ]
        ).T,
        index=unit_id_all,
        columns=[
            "layer",
            "firing_rate",
        ],
    )