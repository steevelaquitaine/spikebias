import numpy as np
import pandas as pd
from collections import Counter
import spikeinterface as si
from spikeinterface import comparison
from matplotlib import pyplot as plt
from src.nodes.load import load_campaign_params


# from src.pipes.validation.old import (
#     run_histo_sorting_errors_neuropixels_2023_02_19 as SortingErrors,
# )

def match_property(
    cell_matching: pd.DataFrame, data_conf: dict, properties: list
):
    """Add cell properties to cell matching dataframe

    Args:
        cell_matching (pd.DataFrame): _description_
        data_conf (dict): _description_
        properties (list): for example ["morph_class", "etype"]

    Returns:
        _type_: _description_
    """

    # get true cells gids
    true_cells_gids = cell_matching["true_cell_match"]

    # get cells properties
    simulation_params = load_campaign_params(data_conf)
    properties = simulation_params["circuit"].cells.get(
        true_cells_gids, properties=properties
    )

    # add to cell_matching dataset
    properties = properties.reset_index().rename(
        columns={"index": "true_cell_match"}
    )
    return cell_matching.merge(properties, on="true_cell_match", how="outer")


# def match_true_and_sorter_cell_firing_rate(data_conf: dict, **vararg: dict):
#     """match true and sorted cells and add their firing rates

#     Args:
#         data_conf (dict): _description_

#     Returns:
#         _type_: _description_
#     """
#     # Get recording duration
#     # get config

#     # get Recording extractor
#     Recording = si.load_extractor(
#         data_conf["preprocessing"]["output"]["trace_file_path"]
#     )

#     # get cell matches b/w ground truth and sorter
#     out = match_sorted_to_true_neuropixels_2023_02_19

#     # get ground truth firing rates
#     TrueSorting = out["GTSortingExtractor"]
#     true_firing_rates = spikestats.get_firing_rates(TrueSorting, Recording)
#     true_firing_rates = true_firing_rates.rename(
#         columns={"firing rate": "true firing rate", "cells": "true_cell_match"}
#     )

#     # get KS3 sorted firing rates
#     KS3Sorting = out["SortingExtractorKS3"]
#     ks3_firing_rates = spikestats.get_firing_rates(KS3Sorting, Recording)
#     ks3_firing_rates = ks3_firing_rates.rename(
#         columns={"firing rate": "ks3 firing rate", "cells": "sorted_cell"}
#     )

#     # get match between true and sorted
#     out = SortingErrors.run(**vararg)
#     cell_matching = out["cell_matching"]

#     # add true cell firing rates
#     cell_matching_new = cell_matching.merge(
#         true_firing_rates, on="true_cell_match", how="outer"
#     )

#     # add sorted cell firing rates
#     cell_matching_new = cell_matching_new.merge(
#         ks3_firing_rates, on="sorted_cell", how="outer"
#     )
#     return cell_matching_new


def get_match_mx(
    true_path: str, sorted_path: str, start_frame: int, end_frame: int, dt:float
):
    """get spikeinterface matching object from the
    comparison.compare_sorter_to_ground_truth module

    Args:
        true_path (_type_): _description_
        sorted_path (_type_): _description_
        start_frame: int, 
        end_frame: int
        dt (float): delta_time

    Returns:
        _type_: _description_
    """
    # load ground truth
    SortingTrue = si.load_extractor(true_path)
    SortingTrue = SortingTrue.frame_slice(start_frame=start_frame, end_frame=end_frame)

    # load sorting extractor
    Sorting = si.load_extractor(sorted_path)
    Sorting = Sorting.frame_slice(start_frame=start_frame, end_frame=end_frame)

    # pairwise matches
    match = comparison.compare_sorter_to_ground_truth(
        SortingTrue, Sorting, exhaustive_gt=True, delta_time=dt
    )
    
    # agreement score between sorted and true units
    return match, SortingTrue, Sorting
    

def reformat_match_mx(Match: si.comparison.paircomparisons.GroundTruthComparison):
    """each row are sorted in descending order. The columns are not
    labelled as the raw ground truth anymore but become labelled
    as best match ground truth to the worst match.
    
    Args:
        Match (si.comparison.paircomparisons.GroundTruthComparison): _description_

    Returns:
        _type_: _description_
    """

    # get sorted x true units' agreement scores
    overmerging_matx = Match.agreement_scores.T

    # sort each row such that the row with the highest 
    # score be first, while column order stays unchanged
    #argmax = overmerging_matx.T.idxmax().to_frame()
    max = overmerging_matx.T.max()
    descending_ix = np.argsort(max)[::-1]
    overmerging_matx_2 = overmerging_matx.iloc[descending_ix]

    # repeat for columns, row order stays auntouched
    #argmax = overmerging_matx_2.idxmax().to_frame()
    max = overmerging_matx_2.max()
    descending_ix = np.argsort(max)[::-1]
    return overmerging_matx_2.iloc[:, descending_ix]


def plot_true_units_quality(axis, biases_count: pd.DataFrame):

    # set colors for combination of biases
    oversplit_plus_overmerged = np.array([[0.6, 0.9, 0.6], [0, 0.7, 1]]).mean(axis=0)
    well_detected_plus_correlated_units_plus_overmerged = np.array(
        [[1, 0, 0], [0, 0.7, 1]]
    ).mean(axis=0)

    # set all colors
    colors = [
        [0.7, 0.1, 0.1],  # "well_detected" (strong red)
        [1, 0, 0],        # "well_detected_plus_correlated_units" (red)
        well_detected_plus_correlated_units_plus_overmerged,
        [1, 0.85, 0.85],  # "poorly_detected" (pink)
        [0, 0.7, 1],  # "overmerged" (green)
        [0.6, 0.9, 0.6],  # "oversplit" (blue)
        oversplit_plus_overmerged,
        [0.95, 0.95, 0.95],  # "below chance"
        "k",  # "missed"
    ]

    # ratios
    biases_ratio = biases_count / biases_count.sum()

    # plot
    ax = (biases_ratio).T.plot.bar(
        ax=axis,
        stacked=True,
        color=colors,
        width=0.7,
        #edgecolor=[0.5, 0.5, 0.5],
        linewidth=0.2,
    )

    # set axis legend
    ax.spines[["left", "right", "top", "bottom"]].set_visible(False)
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)
    ax.set_xticklabels(biases_ratio.columns, rotation=45, ha="right")
    ax.set_ylabel("Sorting biases (ratio)", fontsize=9)
    ax.legend(
        biases_count.index,
        ncol=1,
        loc="lower left",
        bbox_to_anchor=(1, 0),
        frameon=False,
        handletextpad=0.6,
    )
    plt.tight_layout()
    return axis


def create_true_biases_df(true_biases: pd.DataFrame):

    # format dataframe to plot
    bias_types = [
        "good",
        "good + corr",
        "good + corr + overmerged",
        "poor",
        "overmerged",
        "oversplit",
        "oversplit + overmerged",
        "below chance",
        "missed",
    ]

    # count each bias
    count_by_class = dict(Counter(true_biases.values.squeeze().tolist()))

    # fill up count per bias
    for key_k in bias_types:
        try:
            count_by_class[key_k]
        except:
            count_by_class[key_k] = 0

    # order by "bias_types"
    reordered = {k: count_by_class[k] for k in bias_types}

    # create table
    biases_ratio_df = pd.DataFrame(
        {"cell_count": list(reordered.values())}, index=list(reordered.keys())
    )

    # unit-test
    assert biases_ratio_df.sum().values[0] == len(
        true_biases
    ), "true units have not been classified"
    return biases_ratio_df


def classify_true_unit_biases(match_mx:pd.DataFrame, det_thresh:float, chance:float):
    """classify ground truth units' biases with a rule-based classification tree

    Args:
        match_mx (_type_): agreement score matrix b/w ground truth and sorted units
        det_thresh (_type_): good sorting level (typically 0.8)
        chance (float): chance level (typically 0.1)

    Returns:
        _type_: _description_
    """

    # create masks
    mask_above_det = match_mx >= det_thresh
    mask_below_chance = match_mx <= chance
    mask_in_between = np.logical_and(
        match_mx < det_thresh, match_mx > chance
    )
    mask_entirely_missed = match_mx == 0

    # implement tree to classify ground truths
    # find ground truth (cols) with one mask_above_det=True and other mask_below_chance = True

    gt_classes = []
    df = pd.DataFrame()

    # loop over ground truth units
    for gt_i in range(match_mx.shape[1]):

        # check if that ground truth has a single sorted unit
        # with an agreement score above detection threshold
        if any(mask_above_det.iloc[:, gt_i]):

            # get this ground truth detection stata
            is_detected = mask_above_det.iloc[:, gt_i]
            detected_loc = np.where(is_detected)[0]
            detected_ix = is_detected.index[detected_loc]

            # get other cells
            other_cells_ix = is_detected.drop(index=detected_ix).index

            # get this ground truth below chance stata
            is_below_chance = mask_below_chance.iloc[:, gt_i]

            # check if all other sorted units are below chance
            if all(is_below_chance.loc[other_cells_ix]):
                gt_classes.append("good")

            # if another unit has an agreement score
            # above chance level, it is: well detected + correlated unit
            else:
                gt_classes.append("good + corr")

        # case where ground truth matches only one sorted unit
        # with a score b/w detection and chance and
        # other units below chance
        # no score are above detection
        elif (sum(mask_in_between.iloc[:, gt_i]) == 1) and (
            any(mask_above_det.iloc[:, gt_i]) == False
        ):
            gt_classes.append("poor")

        # case a true unit is associated is a sorted unit with score
        # between detection and chance that is associated with other
        # true units with scores between detection and chances
        elif sum(mask_in_between.iloc[:, gt_i]) > 1:
            gt_classes.append("oversplit")

        # check that all sorted units have scores below
        # chance
        elif all(mask_below_chance.iloc[:, gt_i]):
            if all(mask_entirely_missed.iloc[:, gt_i]):
                gt_classes.append("missed")
            else:
                gt_classes.append("below chance")

    # Detect overmerged units and combinations -------------

    # if one of its sorted units with score between
    # detection and chance has also a score between
    # detection and chance with another true unit
    # the true unit is overmerged (with another true unit)
    true_units_loc = np.where(mask_in_between.sum(axis=0) >= 1)[0]
    true_units = mask_in_between.columns[true_units_loc]
    gt_overmerged = dict()

    for gt_i in range(len(true_units_loc)):
        target_true_units_mx = mask_in_between.iloc[:, true_units_loc]
        sorted_u = np.where(target_true_units_mx.iloc[:, gt_i])[0]

        # check overmerged (that sorted unit merges other true units)
        if any(mask_in_between.iloc[sorted_u, :].sum(axis=1) > 1):
            overmerged_bool = mask_in_between.iloc[sorted_u, :].sum(axis=1) > 1
            overmerging_sorted = overmerged_bool.index[
                np.where(overmerged_bool)[0]
            ].to_list()
            gt_overmerged[true_units[gt_i]] = overmerging_sorted

    # what other biases do overmerged units have?
    all_true_units = match_mx.columns
    gt_classes_df = pd.DataFrame(data=gt_classes, index=all_true_units.to_list())
    print(
        "combination of biases:", np.unique(gt_classes_df.loc[gt_overmerged.keys(), :])
    )

    # label combination of biases
    gt_classes_df.loc[gt_overmerged.keys(), :] = gt_classes_df.loc[
        gt_overmerged.keys(), :
    ].apply(lambda x: x + " + overmerged")

    # poorly detected + overmerged units are poorly detected because overmerged so simply overmerged
    gt_classes_df[gt_classes_df == "poor + overmerged"] = "overmerged"

    # unit-test
    assert (
        len(gt_classes_df) == match_mx.shape[1]
    ), "true units have not been classified"
    return gt_classes_df


def compute_true_units_quality(
    recs: list,
    true_paths: list,
    sorted_paths: list,
    sfreqs: list,
    good_thr: float,
    chance_thr: list,
    delta_time: float,
):
    # compute true unit quality for recording(s)
    data_df = pd.DataFrame()
    for ix, rec_nm in enumerate(recs):

        # 1. compute best matches
        Match, _, Sorting = get_match_mx(
            true_paths[ix], sorted_paths[ix], 0, 10 * 60 * sfreqs[ix], delta_time
        )

        # 2. reformat
        match_df = reformat_match_mx(Match)

        # 3. select sorted single unit data
        if "KSLabel" in Sorting.get_property_keys():
            match_df = match_df.loc[
                Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"], :
            ]

        # 4. classify biases
        biases_df = classify_true_unit_biases(match_df, good_thr, chance_thr)

        # 5. calculate biases' ratio
        ratios_df = create_true_biases_df(biases_df)

        # 6. store all data in a dataframe
        data_df[rec_nm] = ratios_df["cell_count"].values
    data_df.index = ratios_df.index
    return data_df


def get_SpikeInterface_matching_object(gt_sorting_path, ks3_sorting_path):
    """get spikeinterface matching object from the
    comparison.compare_sorter_to_ground_truth module

    Args:
        ks3_sorting_path (_type_): _description_
        gt_sorting_path (_type_): _description_

    Returns:
        _type_: _description_
    """
    # load ground truth spikes and units
    GTSortingExtractor = si.load_extractor(gt_sorting_path)

    # load Kilosort3 Sorted spikes and units
    SortingExtractorKS3 = si.load_extractor(ks3_sorting_path)

    # agreement score between sorted and true units
    return comparison.compare_sorter_to_ground_truth(
        GTSortingExtractor, SortingExtractorKS3, exhaustive_gt=True
    )


def sort_matching_mx(MatchingObject):
    """sort true/sorted unit matching score matrix

    Args:
        MatchingObject (_type_): _description_

    Returns:
        (pd.DataFrame): agreement scores for sorted unit ids (rows) x true unit ids (columns)
    """
    # get sorted x true units' agreement scores
    overmerging_matx = MatchingObject.agreement_scores.T

    # sort each row such that the row with the highest score be first, while column order stays unchanged
    argmax = overmerging_matx.T.idxmax().to_frame()
    max = overmerging_matx.T.max()
    descending_ix = np.argsort(max)[::-1]
    overmerging_matx_2 = overmerging_matx.iloc[descending_ix]

    # repeat for columns, row order stays auntouched
    argmax = overmerging_matx_2.idxmax().to_frame()
    max = overmerging_matx_2.max()
    descending_ix = np.argsort(max)[::-1]
    overmerging_matx_2 = overmerging_matx_2.iloc[:, descending_ix]
    return overmerging_matx_2


def get_oversplit_units(agreement_scores: pd.DataFrame):
    """get the true unit ids that are oversplit (with agreement
    above zero for more than one sorted unit)

    Args:
        agreement_score (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    is_oversplit = agreement_scores.astype(bool).sum(axis=1) > 0
    return is_oversplit[is_oversplit].index.tolist()


def get_missed_units(agreement_scores: pd.DataFrame):
    """list the id of missed units (with an agreement score of zero
    will all sorted units)

    Args:
        cell_matching_new (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    is_missed = (agreement_scores.sum(axis=1) > 0) == False
    return is_missed[is_missed].index.tolist()


def get_detected_units(agreement_scores: pd.DataFrame):
    """list the id of detected units (with a non-null agreement score
    with at least one sorted unit)

    Args:
        agreement_scores (pd.DataFrame): matrix of agreement scores

    Returns:
        _type_: _description_
    """
    is_detected = agreement_scores.sum(axis=1) > 0
    return is_detected[is_detected].index.tolist()


def init_CellMatchingObject(agreement_scores: pd.DataFrame):
    """get the best sorted unit's match for each true cell
    for each true cell, we assign as best match the sorted cell
    with the highest agreement score

    Args:
        agreement_scores (pd.DataFrame): _description_

    Returns:
        _type_: pd.DataFrame
    """
    cell_matching = agreement_scores.T.idxmax().to_frame()
    cell_matching.columns = ["sorted_unit"]
    cell_matching.index.name = "true_unit"
    cell_matching = cell_matching.reset_index()

    # set sorted units for missed units to NaN (currently
    # they are set by idxmax to the first sorted unit id (0)
    missed = get_missed_units(agreement_scores)
    cell_matching["sorted_unit"][
        cell_matching["true_unit"].isin(missed)
    ] = np.nan
    return cell_matching


def add_oversplit_units(cell_matching: pd.DataFrame, MatchingObject):
    """add oversplit units to cell_matching object

    Returns:
        _type_: _description_
    """
    cell_matching["oversplit"] = False
    oversplit_units = get_oversplit_units(MatchingObject.agreement_scores)
    cell_matching["oversplit"][
        cell_matching["true_unit"].isin(oversplit_units)
    ] = True
    return cell_matching


def add_missed_units(cell_matching: pd.DataFrame, MatchingObject):
    """add oversplit units to cell_matching object

    Returns:
        _type_: _description_
    """
    cell_matching["missed"] = False
    missed = get_missed_units(MatchingObject.agreement_scores)
    cell_matching["missed"][cell_matching["true_unit"].isin(missed)] = True
    return cell_matching


def match_sorted_to_true_neuropixels_2023_02_19(
    gt_sorting_path: str, ks3_sorting_path: str
):
    # get SpikeInterface's matching object
    MatchingObject = get_SpikeInterface_matching_object(
        gt_sorting_path, ks3_sorting_path
    )

    # get true cell best matches based on max accuracy
    matching = init_CellMatchingObject(MatchingObject.agreement_scores)

    # flag oversplit true units
    matching = add_oversplit_units(matching, MatchingObject)

    # flag missed true units
    matching = add_missed_units(matching, MatchingObject)

    # # Add agreement scores
    # # --------------------
    # # add max agreement score to dataframe
    # sorted_cells = cell_matching["sorted_cell"].dropna().astype("int")
    # max_agreement_scores = []
    # # get the agreement scores of the matched sorted-true pairs
    # for s_i in sorted_cells:
    #     max_agreement_scores.append(
    #         MatchingObject.agreement_scores.loc[
    #             cell_matching["true_cell_match"][s_i], s_i
    #         ]
    #     )
    # # add agreement scores to dataset
    # max_agreement_scores = pd.DataFrame(
    #     max_agreement_scores, columns=["agreement_score"]
    # )
    # cell_matching = cell_matching.join(max_agreement_scores, how="outer")
    # # write to .parquet
    # parent_path = os.path.dirname(CELL_MATCHING_PATH)
    # if not os.path.isdir(parent_path):
    #     os.makedirs(parent_path)
    # cell_matching.to_parquet(CELL_MATCHING_PATH)
    # return {
    #     "cell_matching": matching,
    #     "GTSortingExtractor": GTSortingExtractor,
    #     "SortingExtractorKS3": SortingExtractorKS3,
    #     "MatchingObject": MatchingObject,
    # }
    return {
        "cell_matching": matching,
        "MatchingObject": MatchingObject,
    }
