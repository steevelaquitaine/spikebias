import numpy as np
import pandas as pd
import spikeinterface as si
import spikeinterface.qualitymetrics as quality

from src.nodes.load import load_campaign_params
from src.nodes.postpro import waveform
from src.nodes.metrics import metrics 
# from src.nodes.postpro.cell_matching import (
#     match_true_and_sorter_cell_firing_rate as match_fr,
# )

# from src.nodes.postpro.missed_cells import get_missed_vs_detected
from src.nodes.prepro import preprocess

# from src.pipes.sorting import (
#     match_sorted_to_true_neuropixels_2023_02_19 as match,
# )


def add_p_bias(
    df: pd.DataFrame, low_rate_ceiling: float, mid_rate_ceiling: float
):
    """_summary_

    Args:
        df (pd.DataFrame): cell matching dataframe
        low_rate_ceiling (float): _description_
        mid_rate_ceiling (float): _description_

    Returns:
        _type_: _description_
    """
    feat = pd.DataFrame()

    # biased firing rate range feature
    feat["p_bias"] = df["true firing rate"]

    # flag firing rates with negative probability bias
    feat["p_bias"][df["true firing rate"] < low_rate_ceiling] = "neg_p_bias"

    # flag firing rate with positive probability bias
    feat["p_bias"][
        np.logical_and(
            df["true firing rate"] >= low_rate_ceiling,
            df["true firing rate"] < mid_rate_ceiling,
        )
    ] = "pos_p_bias"

    # flag firing rate with no probability bias
    feat["p_bias"][df["true firing rate"] >= mid_rate_ceiling] = "no_p_bias"
    df["p_bias"] = feat["p_bias"]
    return df


def add_firing_rates(
    cell_matching: pd.DataFrame,
    gt_sorting_path: str,
    ks3_sorting_path: str,
    data_conf: dict,
):
    # get true and sorted firing rates
    (true_firing_rates, ks3_firing_rates) = get_true_and_sorted_firing_rates(
        gt_sorting_path, ks3_sorting_path, data_conf
    )

    # add true units firing rates
    cell_matching_new = cell_matching.merge(
        true_firing_rates, on="true_unit", how="outer"
    )

    # add sorted units firing rates
    cell_matching_new = cell_matching_new.merge(
        ks3_firing_rates, on="sorted_unit", how="outer"
    )
    return cell_matching_new


def get_true_and_sorted_firing_rates(
    gt_sorting_path: str,
    ks3_sorting_path: str,
    data_conf: dict,
):
    # get Recording extractor
    Recording = si.load_extractor(
        data_conf["preprocessing"]["output"]["trace_file_path"]
    )

    # load ground truth spikes and units
    GTSortingExtractor = si.load_extractor(gt_sorting_path)

    # load Kilosort3 Sorted spikes and units
    SortingExtractorKS3 = si.load_extractor(ks3_sorting_path)

    # get firing rates
    true_firing_rates = metrics.get_firing_rates(
        GTSortingExtractor, Recording
    )
    ks3_firing_rates = metrics.get_firing_rates(
        SortingExtractorKS3, Recording
    )
    # format
    true_firing_rates = true_firing_rates.rename(
        columns={"firing rate": "true firing rate", "cells": "true_unit"}
    )
    ks3_firing_rates = ks3_firing_rates.rename(
        columns={"firing rate": "ks3 firing rate", "cells": "sorted_unit"}
    )
    return true_firing_rates, ks3_firing_rates


def test_add_firing_rates():
    """unit-test for "add_firing_rates" function"""
    # create mock cell matching
    cell_matching = pd.DataFrame(
        data=np.array([[0, 1, 2, 3, 4], [14, 13, 12, 11, 10]]).T,
        columns=["true_unit", "sorted_unit"],
    )

    # create mock true and sorted firing rates
    true_firing_rates = pd.DataFrame(
        data=np.array([[0, 1, 2, 3, 4], [0.1, 0.2, 0.3, 0.4, 0.5]]).T,
        columns=["true_unit", "true firing rate"],
    )
    ks3_firing_rates = pd.DataFrame(
        data=np.array([[10, 11, 12, 13, 14], [1, 2, 3, 4, 5]]).T,
        columns=["sorted_unit", "ks3 firing rate"],
    )

    # add true units firing rates
    cell_matching_new = cell_matching.merge(
        true_firing_rates, on="true_unit", how="outer"
    )

    # add sorted units firing rates
    cell_matching_new = cell_matching_new.merge(
        ks3_firing_rates, on="sorted_unit", how="outer"
    )

    # test expected results
    assert all(
        cell_matching_new["true firing rate"]
        == np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    ), "true firing rates are wrong"
    assert all(
        cell_matching_new["ks3 firing rate"] == np.array([5, 4, 3, 2, 1])
    ), "sorted firing rates are wrong"


def add_firing_rate_change(df: pd.DataFrame):
    """_summary_

    Args:
        df (pd.DataFrame): cell matching dataframe

    Returns:
        _type_: _description_
    """
    feat = pd.DataFrame()

    # "rate change" feature
    rate_change = df["ks3 firing rate"] - df["true firing rate"]
    feat["rate_change_feat"] = rate_change
    feat["rate_change_feat"][rate_change > 0] = "overestimated"
    feat["rate_change_feat"][rate_change < 0] = "underestimated"
    feat["rate_change_feat"][rate_change == 0] = "same"
    df["rate_change_feat"] = feat["rate_change_feat"]
    return df


def add_false_positive_spikes_count(df: pd.DataFrame, Match):
    """_summary_

    notes: As stated in SpikeInterface's source code, 'Labels (TP, FP, FN)
    can be computed only with hungarian match'

    Args:
        df (pd.DataFrame): _description_
        Match (_type_): we call its `get_labels2()` method

    Returns:
        _type_: _description_

    Reference:
        https://spikeinterface.readthedocs.io/en/latest/modules/comparison.html
        #more-information-about-hungarian-or-best-match-methods
    """
    n_fp = []
    # for u_i in df["sorted_cell"]:
    #     if not np.isnan(u_i):
    #         n_fp.append(
    #             sum(MatchinObject.get_labels2(int(u_i))[0] == "FP")
    #         )
    #     else:
    #         n_fp.append(np.nan)
    # df["fp_spike_count"] = n_fp
    for ix in range(df.shape[0]):
        sort_i = df["sorted_cell"].iloc[ix]
        true_i = df["true_cell_match"].iloc[ix]
        if not np.isnan(sort_i):
            n_tp = Match.match_event_count.loc[true_i, sort_i]
            n_fp.append(Match.event_counts2[sort_i] - n_tp)
        else:
            n_fp.append(np.nan)
    df["fp_spike_count"] = n_fp
    return df


def add_missed_spikes_count(df: pd.DataFrame, Match):
    """_summary_

    notes: As stated in SpikeInterface's source code, 'Labels (TP, FP, FN)
    can be computed only with hungarian match'

    Args:
        df (pd.DataFrame): _description_
        Match (_type_): we call its `Match.event_counts1()` method
        which must correspond to ground truth spike count

    Returns:
        pd.DataFrame: _description_
    """
    n_fn = []
    # for u_i in df["sorted_cell"]:
    #     if not np.isnan(u_i):
    #         n_fn.append(
    #             sum(Match.get_labels2(int(u_i))[0] == "FN")
    #         )
    #     else:
    #         n_fn.append(np.nan)
    # df["fn_spike_count"] = n_fn
    for ix in range(df.shape[0]):
        sort_i = df["sorted_cell"].iloc[ix]
        true_i = df["true_cell_match"].iloc[ix]
        if not np.isnan(sort_i):
            n_tp = Match.match_event_count.loc[true_i, sort_i]
            n_fn.append(Match.event_counts1[true_i] - n_tp)
        else:
            n_fn.append(np.nan)
    df["fn_spike_count"] = n_fn
    return df


def add_property(
    cell_matching: pd.DataFrame, data_conf: dict, properties: list
):
    """Add cell properties to cell matching dataframe

    Args:
        cell_matching (pd.DataFrame): _description_
        data_conf (dict): _description_
        properties (list): for example ["morph_class", "etype"]

    Returns:
        pd.DataFrame: matching between true and sorted units with
        true unit layer
    """

    # get true cells gids
    true_cells_gids = cell_matching["true_unit"]

    # get cells properties
    simulation_params = load_campaign_params(data_conf)
    properties = simulation_params["circuit"].cells.get(
        true_cells_gids, properties=properties
    )

    # add to cell_matching dataset
    properties = properties.reset_index().rename(
        columns={"index": "true_unit"}
    )
    return cell_matching.merge(properties, on="true_unit", how="outer")


# def add_true_spike_count(cell_matching: pd.DataFrame, data_conf: dict):
#     """Add true cell spike count to cell matching dataframe

#     Args:
#         cell_matching (pd.DataFrame): _description_
#         data_conf (dict): _description_

#     Returns:
#         _type_: _description_
#     """

#     # get cell matches b/w ground truth and sorter
#     out = match.run()

#     # get ground truth spike count
#     TrueSorting = out["GTSortingExtractor"]
#     true_spike_count = spike_stats.get_spike_count(TrueSorting)
#     true_spike_count = true_spike_count.rename(
#         columns={"total spike": "true spike count", "cells": "true_cell_match"}
#     )
#     return cell_matching.merge(
#         true_spike_count, on="true_cell_match", how="outer"
#     )


def add_true_spike_count(cell_matching: pd.DataFrame, data_conf: dict):
    """Add true cell spike count to cell matching dataframe

    Args:
        cell_matching (pd.DataFrame): _description_
        data_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # set ground truth spikes and cells path
    GT_SORTING_PATH = data_conf["sorting"]["simulation"]["ground_truth"][
        "output"
    ]

    # get ground truth spike count
    TrueSorting = si.load_extractor(GT_SORTING_PATH)
    true_spike_count = metrics.get_spike_count(TrueSorting)
    true_spike_count = true_spike_count.rename(
        columns={"total spike": "true spike count", "cells": "true_unit"}
    )
    return cell_matching.merge(true_spike_count, on="true_unit", how="outer")


def test_add_true_spike_count():
    """unit-testing"""

    # create mock cell matching
    cell_matching = pd.DataFrame(
        data=np.array([[0, 1, 2, 3, 4], [14, 13, 12, 11, 10]]).T,
        columns=["true_unit", "sorted_unit"],
    )

    # create mock spike counts
    true_spike_count = pd.DataFrame(
        data=np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]]).T,
        columns=["true_unit", "true spike count"],
    )

    # add spike count
    cell_matching = cell_matching.merge(
        true_spike_count, on="true_unit", how="outer"
    )
    assert all(
        cell_matching["true spike count"] == np.array([0, 1, 2, 3, 4, 5])
    ), "spike counts are wrongly merged"


def add_sorted_spike_count(cell_matching: pd.DataFrame, data_conf: dict):
    """Add sorted cell spike count to cell matching dataframe

    Args:
        cell_matching (pd.DataFrame): _description_
        data_conf (dict): _description_

    Returns:
        _type_: _description_
    """

    # set Kilosort sorted spikes and cells path
    KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]

    # load Kilosort3 Sorted spikes and units
    SortingExtractorKS3 = si.load_extractor(KS3_SORTING_PATH)
    sorted_spike_count = metrics.get_spike_count(SortingExtractorKS3)
    sorted_spike_count = sorted_spike_count.rename(
        columns={"total spike": "sorted spike count", "cells": "sorted_unit"}
    )
    return cell_matching.merge(
        sorted_spike_count, on="sorted_unit", how="outer"
    )


def test_add_sorted_spike_count():
    """unit-test"""
    # create mock cell matching
    cell_matching = pd.DataFrame(
        data=np.array([[0, 1, 2, 3, 4], [14, 13, 12, 11, 10]]).T,
        columns=["true_unit", "sorted_unit"],
    )

    # create mock spike counts
    sorted_spike_count = pd.DataFrame(
        data=np.array(
            [
                [14, 13, 12, 11, 10, 9, 8, 7, 6, 5],
                [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            ]
        ).T,
        columns=["sorted_unit", "sorted spike count"],
    )

    # add spike count
    cell_matching = cell_matching.merge(
        sorted_spike_count, on="sorted_unit", how="outer"
    )
    assert all(
        cell_matching["sorted spike count"]
        == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    ), "sorted spike counts are wrongly merged"


def add_true_snr(cell_matching: pd.DataFrame, data_conf: dict):
    """_summary_

    Args:
        cell_matching (pd.DataFrame): _description_
        data_conf (dict): _description_
    """
    # get parameters
    STUDY_FOLDER = data_conf["postprocessing"]["waveform"]["study"]

    # load recording
    recording = preprocess.load(data_conf=data_conf)

    # load waveform extractor (a "study" must have been created with waveform.run() pipeline)
    WaveformExtractor = waveform.load(recording, study_folder=STUDY_FOLDER)

    # calculate snrs, dict of cell_id:snr ({19690: 29.467033, 24768: 92.0003 ...)
    snrs = quality.compute_snrs(WaveformExtractor)

    # convert to dataframe
    snrs = pd.DataFrame.from_dict(snrs, orient="index")

    # set dataframe column names
    snrs = snrs.rename(columns={0: "snr"})
    snrs = snrs.reset_index().rename(columns={"index": "true_cell_match"})
    return cell_matching.merge(snrs, on="true_cell_match", how="outer")


# def add_is_missed(cell_matching: pd.DataFrame, data_conf: dict):
#     """label missed and detected cells

#     Args:
#         cell_matching (pd.DataFrame): _description_
#         data_conf (dict): _description_

#     Returns:
#         _type_: _description_
#     """
#     # add firing rates
#     cell_matching = match_fr(data_conf, no_plot=True)

#     # detect missed and detected true cells
#     missed_cells, detected_cells = get_missed_vs_detected(
#         cell_matching
#     ).values()

#     # label missed and detected cells
#     cell_matching.loc[missed_cells.index, "detection_label"] = "missed"
#     cell_matching.loc[detected_cells.index, "detection_label"] = "detected"
#     return cell_matching
