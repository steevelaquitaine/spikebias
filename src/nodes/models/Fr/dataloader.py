"""nodes for the models

author: steeve.laquitaine@epfl.ch

Returns:
    _type_: _description_
"""

from itertools import combinations, product
import pandas as pd
import spikeinterface as si
from spikeinterface import qualitymetrics as qm
import spikeinterface.core.template_tools as ttools
import logging
import logging.config
from time import time
import yaml
import itertools
import numpy as np

# custom package
from src.nodes.metrics.quality import get_scores
from src.nodes.metrics import metrics

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")


def get_accuracy(Sorting, SortingTrue, delta_time=1.3):
    scores = get_scores(Sorting, SortingTrue, delta_time)
    return scores.max(axis=0)


def get_snrs(study_path: str):

    # load WaveformExtractor
    We = si.WaveformExtractor.load_from_folder(study_path)

    # signal-to-noise ratio
    snrs = qm.compute_snrs(
        We, peak_sign="neg", peak_mode="extremum", unit_ids=We.sorting.unit_ids
    )
    return snrs.values()


def set_synapse_class(dataset: pd.DataFrame, SortingTrue):
    """Add synapse class (EXC, INT) to dataset
    """
        
    # unit-test
    assert "synapse_class" in SortingTrue.get_property_keys(), """synapse_class ...
    metadata is missing, you must run sorting pipe 
    for ground truth"""
    
    # one-hot-encode synapse class
    dataset["synapse_class"] = pd.get_dummies(
        SortingTrue.get_property("synapse_class")
    )["EXC"].values
    return dataset


def load_dataset(sort_path: str, sorttrue_path: str, recording_path: str, study_path:str, duration_sec=600, n_min_spikes=10):
    """create dataset with predictor features
    and target sorting accuracy

    Args:
        sort_path
        sorttrue_path (SpikeInterface Sorting Extractor): Sorting extractor
        with units metadata as saved properties

    Returns:
        _type_: _description_
    """
    # load SortingExtractors
    Sorting = si.load_extractor(sort_path)
    SortingTrue = si.load_extractor(sorttrue_path)

    # create dataset
    dataset = pd.DataFrame()

    # set firing rate
    dataset["firing_rate"] = [
        SortingTrue.get_total_num_spikes()[unit] / duration_sec
        for unit in SortingTrue.get_total_num_spikes()
    ]
        
    # set synapse class
    dataset = set_synapse_class(dataset, SortingTrue)

    # set etype
    etype_df = pd.get_dummies(SortingTrue.get_property("etype"))
    dataset = pd.merge(dataset, etype_df, left_index=True, right_index=True)

    # set layer
    layers = SortingTrue.get_property("layer")
    layers = np.array(["2/3" if (w == "2") or (w == "3") else w for w in layers])
    feature_layer_df = pd.get_dummies(layers)
    feature_layer_df.columns = [
        "layer_1",
        "layer_23",
        "layer_4",
        "layer_5",
        "layer_6",
    ]
    dataset = pd.merge(dataset, feature_layer_df, left_index=True, right_index=True)

    # set distance to nearest site
    dataset["distance"] = metrics.get_true_unit_true_distance_to_nearest_site(
        SortingTrue, recording_path
    )
    
    # set snr
    dataset["snr"] = get_snrs(study_path)

    # set spike spatial extent
    spatial_extent = metrics.get_spatial_spread_all_units(recording_path, study_path)
    dataset["spatial_extent"] = [
        spatial_extent[unit]
        for unit in spatial_extent
    ]
    
    # set exc_mini_frequency (5 unique values)
    dataset["exc_mini_frequency"] = SortingTrue.get_property("exc_mini_frequency")

    # set dynamics_holding_current
    dataset["dynamics_holding_current"] = SortingTrue.get_property(
        "dynamics_holding_current"
    )

    # set dynamics_holding_current
    dataset["model_template"] = SortingTrue.get_property("model_template")
        
    # add target: sorting accuracy
    dataset["sorting_accuracy"] = get_accuracy(
        Sorting, SortingTrue, delta_time=1.3
    ).values

    # set unit ids as indices
    dataset.index = SortingTrue.unit_ids
    
    # curate units
    logger.info("CURATION ----------------------------")
    
    # filter units with a minimum number of spikes
    logger.info(f"nb of units before filtering by nb of spike: {len(dataset)}")
    dataset[dataset["firing_rate"] * duration_sec >= n_min_spikes]
    logger.info(f"nb after: {len(dataset)}")
    
    # filter units without infinite-value feature
    infdata = dataset.index[dataset.sum(axis=1) == np.inf]
    logger.info(f"nb of units before filtering units with inf feature: {len(dataset)}")
    dataset = dataset.drop(index=infdata)
    logger.info(f"nb of units after: {len(dataset)}")
    return dataset