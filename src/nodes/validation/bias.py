"""Node library to calculate and plot biases

Returns:
    _type_: _description_
"""
import logging
import logging.config
import os
import shutil
from datetime import datetime
from time import time
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import yaml
from matplotlib import pyplot as plt

from src.nodes.dataeng.silico import probe_wiring, recording
from src.nodes.load import load_campaign_params
from src.nodes.postpro import spikestats
from src.nodes.prepro import preprocess
from src.nodes.study import bias
from src.nodes.truth.silico import ground_truth
from src.nodes.utils import get_config, write_metadata

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")



def format_agreement_matrix(MatchingObject):

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
    return overmerging_matx_2.iloc[:, descending_ix]


def classify_true_unit_biases(overmerging_matx_2, det_thresh, chance):

    # create masks
    mask_above_det = overmerging_matx_2 >= det_thresh
    mask_below_chance = overmerging_matx_2 <= chance
    mask_in_between = np.logical_and(
        overmerging_matx_2 < det_thresh, overmerging_matx_2 > chance
    )
    mask_entirely_missed = overmerging_matx_2 == 0

    # implement tree to classify ground truths
    # find ground truth (cols) with one mask_above_det=True and other mask_below_chance = True

    gt_classes = []
    df = pd.DataFrame()

    # loop over ground truth units
    for gt_i in range(overmerging_matx_2.shape[1]):

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
                gt_classes.append("well detected")

            # if another unit has an agreement score
            # above chance level, it is: well detected + correlated unit
            else:
                gt_classes.append("well detected, correlated")

        # case where ground truth matches only one sorted unit
        # with a score b/w detection and chance and
        # other units below chance
        # no score are above detection
        elif (sum(mask_in_between.iloc[:, gt_i]) == 1) and (
            any(mask_above_det.iloc[:, gt_i]) == False
        ):
            gt_classes.append("poorly detected")

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
    all_true_units = overmerging_matx_2.columns
    gt_classes_df = pd.DataFrame(data=gt_classes, index=all_true_units.to_list())
    print(
        "combination of biases:", np.unique(gt_classes_df.loc[gt_overmerged.keys(), :])
    )

    # label combination of biases
    gt_classes_df.loc[gt_overmerged.keys(), :] = gt_classes_df.loc[
        gt_overmerged.keys(), :
    ].apply(lambda x: x + ", overmerged")

    # poorly detected + overmerged units are poorly detected because overmerged so simply overmerged
    gt_classes_df[gt_classes_df == "poorly detected, overmerged"] = "overmerged"
    return gt_classes_df


def create_true_biases_df(true_biases_series):

    # format dataframe to plot
    bias_types = [
        "well detected",
        "well detected, correlated",
        "well detected, correlated, overmerged",
        "poorly detected",
        "overmerged",
        "oversplit",
        "oversplit, overmerged",
        "below chance",
        "missed",
    ]

    # count each bias
    count_by_class = dict(Counter(true_biases_series.values.squeeze().tolist()))

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
    return biases_ratio_df


def plot_biases(axis, biases_count: pd.DataFrame):

    # set colors for combination of biases
    oversplit_plus_overmerged = np.array([[0.6, 0.9, 0.6], [0, 0.7, 1]]).mean(axis=0)
    well_detected_plus_correlated_units_plus_overmerged = np.array(
        [[1, 0, 0], [0, 0.7, 1]]
    ).mean(axis=0)

    # set all colors
    colors = [
        [0.7, 0.1, 0.1],  # "well_detected" (strong red)
        [1, 0, 0],  # "well_detected_plus_correlated_units" (red)
        well_detected_plus_correlated_units_plus_overmerged,
        [1, 0.85, 0.85],  # "poorly_detected" (pink)
        [0, 0.7, 1],  # "overmerged" (green)
        [0.6, 0.9, 0.6],  # "oversplit" (blue)
        oversplit_plus_overmerged,
        [0.95, 0.95, 0.95],  # "below chance"
        "k",  # "missed"
    ]

    biases_ratio = biases_count / biases_count.sum()

    # plot
    ax = (biases_ratio).T.plot.bar(
        ax=axis,
        stacked=True,
        color=colors,
        width=0.9,
        edgecolor=[0.5, 0.5, 0.5],
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


def plot_heatmap(overmerging_matx_2):

    # plot
    # fig, axis = plt.subplots(1, 1, figsize=(2, 10))

    # plot agreement matrix
    mx_to_plot = overmerging_matx_2.iloc[:500, :500].values
    fig, axis = plt.subplots(figsize=(6, 4))

    ax = sns.heatmap(
        mx_to_plot,
        cmap="jet",
        cbar_kws={"shrink": 0.5},
        yticklabels=False,
        xticklabels=False,
    )
    plt.xlabel("true units")
    plt.ylabel("sorted units")
    ax.set_aspect("equal")


def classify_sorted_unit_biases(agreem_mx):

    # note: with this approach (BEST matching approach), the same sorted unit can be paired with more than one true unit
    # we only keep the pairings with highest agreement scores
    # true-sorted unit pairing
    pairing = agreem_mx.T.idxmax(axis=1)
    pairing = pairing.to_frame()
    pairing.columns = ["sorted"]

    # add agreement score
    accuracy = agreem_mx.T.max(axis=1)
    pairing["accuracy"] = accuracy

    # check if the only sorted unit paired with this true unit
    sorted_ids = agreem_mx.index

    df = copy.copy(pairing.iloc[0, :].to_frame().T)
    false_positives = []

    # else keep the pairing with highest agreement score
    # loop over all sorted single unit units
    for ix in range(len(sorted_ids)):
        # case the sorted unit was paired with a ground truth unit
        if any(pairing["sorted"] == sorted_ids[ix]):
            sorted_pairings = pairing[pairing["sorted"] == sorted_ids[ix]].sort_values(
                by="accuracy", ascending=False
            )
            # take max pairing (first row)
            df = pd.concat([df, sorted_pairings.iloc[0, :].to_frame().T])
        else:
            # case the sorted unit was paired with none of the ground truth units
            false_positives.append(sorted_ids[ix])

    df = df[1:]
    df["sorted"] = df["sorted"].astype(int)

    # count biases
    n_good = sum(df["accuracy"] >= DET_THRESH)
    n_poor = sum((df["accuracy"] >= CHANCE_THRESH) & (df["accuracy"] < DET_THRESH))
    n_below_chance = sum((df["accuracy"] > 0) & (df["accuracy"] < CHANCE_THRESH))
    n_false_pos = len(false_positives)

    # sanity check
    assert n_good + n_poor + n_below_chance + n_false_pos == len(
        sorted_ids
    ), "They must match"
    return {
        "n_good": n_good,
        "n_poor": n_poor,
        "n_below_chance": n_below_chance,
        "n_false_pos": n_false_pos,
    }





def get_recording(
    data_conf: dict,
    param_conf: dict,
    create_recording: bool,
    load_raw_recording: bool,
    prep_recording: bool,
):
    """get SpikeInterface Recording Extractor object

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_
        create_recording (bool): _description_
        load_raw_recording (bool): _description_
        prep_recording (bool): _description_

    Returns:
        _type_: _description_
    """
    # set Recording and Sorting Extractors paths
    if param_conf["run"]["recording"] == "raw":
        RECORDING_PATH = data_conf["recording"]["output"]
    else:
        RECORDING_PATH = data_conf["preprocessing"]["output"][
            "trace_file_path"
        ]

    # get recording
    if create_recording:
        t0 = time()

        # log
        logger.info("casting as SpikeInterface Recording Extractor ...")

        # cast as SpikeInterface Recording object ...
        # and wire the configured probe
        trace_recording = probe_wiring.run(data_conf, param_conf)
        recording.write(trace_recording, data_conf)

        # log
        logger.info(
            "casting as SpikeInterface Recording Extractor - done in %s",
            round(time() - t0, 1),
        )

    if load_raw_recording:
        t0 = time()

        # log
        logger.info("loading SpikeInterface Recording Extractor ...")

        trace_recording = probe_wiring.load(data_conf)

        # log
        logger.info(
            "loading SpikeInterface Recording Extractor - done in %s",
            round(time() - t0, 1),
        )

    if prep_recording:
        t0 = time()
        # log
        logger.info("preprocessing and saving recording ...")

        # preprocess recording
        trace_recording = preprocess.run(data_conf, param_conf)

        # log
        logger.info(
            "preprocessing and saving recording - done: %s",
            round(time() - t0, 1),
        )

    else:
        t0 = time()
        # log
        logger.info("loading SpikeInterface Recording object ...")

        # extract recording
        trace_recording = si.load_extractor(RECORDING_PATH)

        # log
        logger.info(
            "loading SpikeInterface recording - done in %s",
            round(time() - t0, 1),
        )
    return trace_recording


def get_ground_truth(data_conf: dict, param_conf: dict):
    """get SpikeInterface ground truth Sorting Extractor object

    Args:
        data_conf (dict): _description_
        param_conf (dict): _description_

    Returns:
        (): SpikeInterface Sorting Extractor
    """

    # set path
    GT_SORTING_PATH = data_conf["sorting"]["simulation"]["ground_truth"][
        "output"
    ]

    t0 = time()

    # get SpikeInterface ground truth Sorting Extractor object
    if os.path.isdir(GT_SORTING_PATH):
        # log
        logger.info("loading already processed true sorting ...")

        # ground truth spikes
        GtSorting = si.load_extractor(GT_SORTING_PATH)

        # log
        logger.info(
            "loading already processed true sorting - done in %s",
            round(time() - t0, 1),
        )
    else:
        # log
        logger.info("loading ground truth sorting ...")

        # load campaign parameters
        simulation = load_campaign_params(data_conf)

        # get ground truth spikes
        GtSorting = ground_truth.run(simulation, data_conf, param_conf)[
            "ground_truth_sorting_object"
        ]

        # log
        logger.info(
            "loading true sorting - done in %s",
            round(time() - t0, 1),
        )

        # log
        t0 = time()
        logger.info("saving ...")

        # save sorting extractor
        GtSorting.save(folder=GT_SORTING_PATH)

        # log
        logger.info(
            "saving - done in %s",
            round(time() - t0, 1),
        )
    return GtSorting


def sort_with_KS2(
    trace_recording,
    data_conf: dict,
    load: bool,
):
    # sort spikes and cells with Kilosort2
    KILOSORT2_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort2"][
        "output"
    ]

    ss.Kilosort2Sorter.set_kilosort2_path(
        data_conf["sorting"]["sorters"]["kilosort2"]["input"]
    )

    if load:
        t0 = time()

        # log
        logger.info("loading kilosort2 sorting ...")

        # load SpikeInterface's Sorting Extractor
        sorting_KS2 = si.load_extractor(KILOSORT2_SORTING_PATH)

        # log
        logger.info(
            "loading kilosort2 sorting - done in %s",
            round(time() - t0, 1),
        )
    else:
        t0 = time()
        # log
        logger.info("running kilosort2 sorting ...")

        # run sorting
        sorting_KS2 = ss.run_kilosort2(trace_recording, verbose=True)

        # log
        logger.info(
            "running kilosort2 sorting - done in %s",
            round(time() - t0, 1),
        )

        # log
        t0 = time()
        logger.info("saving kilosort2 sorting ...")

        # save Sorting Extractor
        sorting_KS2.save(folder=KILOSORT2_SORTING_PATH)

        # log
        logger.info(
            "saving kilosort2 sorting - done in %s",
            round(time() - t0, 1),
        )
    return sorting_KS2


def sort_with_KS3(
    trace_recording,
    data_conf: dict,
    load: bool,
):
    """sort with Kilosort3 default parameters

    Args:
        trace_recording (_type_): _description_
        data_conf (dict): _description_
        load (bool): _description_

    Returns:
        _type_: _description_
    """

    # set sorting path
    KILOSORT3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"][
        "output"
    ]

    # sort spikes and cells with Kilosort3
    ss.Kilosort3Sorter.set_kilosort3_path(
        data_conf["sorting"]["sorters"]["kilosort3"]["input"]
    )
    t0 = time()

    if load:
        # run sorting
        sorting_KS3 = si.load_extractor(KILOSORT3_SORTING_PATH)

        # log
        logger.info(
            "loading kilosort3 sorting - done in %s",
            round(time() - t0, 1),
        )
    else:
        # run sorting
        sorting_KS3 = ss.run_kilosort3(trace_recording, verbose=True)

        # log
        logger.info(
            "running kilosort3 sorting - done in %s",
            round(time() - t0, 1),
        )
        t0 = time()

        # remove the path
        shutil.rmtree(KILOSORT3_SORTING_PATH, ignore_errors=True)

        # save a new directory
        sorting_KS3.save(folder=KILOSORT3_SORTING_PATH)
        logger.info(
            "write sorting done in %s",
            round(time() - t0, 1),
        )
    return sorting_KS3


def sort_with_KS3_no_minfr(
    trace_recording,
    data_conf: dict,
    load: bool,
    minFR=0,
    minfr_goodchannels=0,
):
    """sort with Kilosort3 witthout minimum firing rate (minFR=0)
    for unit cluster clusters and minimum firing rate per channel
    (minfr_goodchannels=0)

    Args:
        trace_recording (_type_): _description_
        data_conf (dict): _description_
        load (bool): _description_
        minFR (int, optional): _description_. Defaults to 0.
        minfr_goodchannels (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # set sorting path
    KILOSORT3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"][
        "output_no_minfr"
    ]

    # sort spikes and cells with Kilosort3
    ss.Kilosort3Sorter.set_kilosort3_path(
        data_conf["sorting"]["sorters"]["kilosort3"]["input"]
    )
    t0 = time()

    # load sorting
    if load:
        sorting_KS3 = si.load_extractor(KILOSORT3_SORTING_PATH)

        # log
        logger.info(
            "load sorting done in %s",
            round(time() - t0, 1),
        )

    # sort
    else:
        sorting_KS3 = ss.run_kilosort3(
            trace_recording,
            verbose=True,
            minFR=minFR,
            minfr_goodchannels=minfr_goodchannels,
        )

        # log
        logger.info(
            "run sorting done in %s",
            round(time() - t0, 1),
        )
        t0 = time()

        # remove the path
        shutil.rmtree(KILOSORT3_SORTING_PATH, ignore_errors=True)

        # save
        sorting_KS3.save(folder=KILOSORT3_SORTING_PATH)
        logger.info(
            "write sorting done in %s",
            round(time() - t0, 1),
        )
    return sorting_KS3


def plot_firing_rates(title: str, data_conf: dict, df_to_plot: pd.DataFrame):
    """plot ground truth and sorter's firing rates and density difference between them

    Args:
        title (str): _description_
        data_conf (dict): _description_
        df_to_plot (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    # set parameters
    BANDWITH = 1e-1

    # Set figure path
    FIG_PATH = data_conf["figures"]["silico"]["population_bias_silico"]

    # plot
    fig, (ax0, ax1) = plt.subplots(
        1, 2, figsize=(4, 2.8), gridspec_kw={"width_ratios": [3, 1.5]}
    )

    # set colors
    my_colors = {"ground_truth": "w", "kilosort3": "w"}

    # set y-axis legend
    df_to_plot = df_to_plot.rename(
        columns={"firing rate": "firing rate (spikes/sec)"}
    )

    # plot firing rate norm (kernel) densities by sorter
    # --------------------------------------------------
    sns.violinplot(
        data=df_to_plot,
        y="firing rate (spikes/sec)",
        x="sorter",
        ax=ax0,
        palette=my_colors,
        bw=BANDWITH,
        linewidth=1,
        color=[0.2, 0.2, 0.2],
        inner="box",
        width=1.5,
        scale="area",
    )

    # color inner boxplots
    ax0.get_children()[1].set_color("k")
    ax0.get_children()[2].set_color("k")
    ax0.get_children()[5].set_color("k")
    ax0.get_children()[6].set_color("k")

    # detect outliers
    # ground truth outliers
    true_firing = df_to_plot["firing rate (spikes/sec)"][
        df_to_plot["sorter"] == "ground_truth"
    ].values
    q1_gt, q3_gt = np.percentile(true_firing, [25, 75])
    whisker_low_gt = q1_gt - (q3_gt - q1_gt) * 1.5
    whisker_high_gt = q3_gt + (q3_gt - q1_gt) * 1.5
    outliers_gt = true_firing[
        (true_firing > whisker_high_gt) | (true_firing < whisker_low_gt)
    ]

    # kilosort3 outliers
    ks3_firing = df_to_plot["firing rate (spikes/sec)"][
        df_to_plot["sorter"] == "kilosort3"
    ].values
    q1_ks, q3_ks = np.percentile(ks3_firing, [25, 75])
    whisker_low_ks = q1_ks - (q3_ks - q1_ks) * 1.5
    whisker_high_ks = q3_ks + (q3_ks - q1_ks) * 1.5
    outliers_ks3 = ks3_firing[
        (ks3_firing > whisker_high_ks) | (ks3_firing < whisker_low_ks)
    ]

    # plot outliers
    sns.scatterplot(
        y=outliers_gt,
        x=0,
        marker="o",
        s=15,
        edgecolors="k",
        color="k",
        ax=ax0,
    )
    sns.scatterplot(
        y=outliers_ks3,
        x=1,
        marker="o",
        s=15,
        edgecolors="k",
        color="k",
        ax=ax0,
    )
    sns.despine(fig, top=True, left=False, right=True)

    # set title legend
    ax0.set_title(title, fontsize=9)

    # set y_lim
    min_fr = -0.1
    max_fr = df_to_plot["firing rate (spikes/sec)"].max() + 0.1
    ax0.set_ylim([min_fr, max_fr])
    ax0.xaxis.label.set_size(9)
    ax0.yaxis.label.set_size(9)
    ax0.tick_params(axis="both", which="major", labelsize=9)

    # plot bias
    # ----------
    bias, firing_rates = get_bias_as_fr_proba_change(true_firing, ks3_firing)

    # plot
    ax1.plot(bias, firing_rates, "r")
    ax1.vlines(
        x=0,
        ymin=min(firing_rates),
        ymax=max(firing_rates),
        linestyle=":",
        color="k",
    )
    ax1.set_xlabel(r"sorted-true" "\n" r"(probability change)", fontsize=9)
    ax1.set_title("bias", fontsize=9)
    ax1.set_ylim([min_fr, max_fr])
    ax1.tick_params(axis="both", which="major", labelsize=9)

    plt.tight_layout()

    # create figure path if it does not exist
    parent_path = os.path.dirname(FIG_PATH)
    if not os.path.isdir(parent_path):
        os.makedirs(parent_path)
        logger.info("The figure's path did not exist and was created.")

    # save figure
    fig.savefig(FIG_PATH + ".png")
    return {"fig", fig}


def get_bias_as_fr_proba_change(
    true_firing: np.array, sorter_firing: np.array
):
    """calculate bias between two arrays of firing rates as the differences
    in their kernel density estimates normalized to probabilities.
    The bias is the change in the probabilities of observing each
    firing rate.

    note: we chose scipy.stats.gaussian_kde because it can
    evaluate the densities on a common firing rate support from
    datasets with different sample sizes and ranges of firing rates

    Args:
        true_firing (np.array): ground truth firing rates
        sorter_firing (np.array): a sorter's firing rates

    Returns:
        _type_: _description_
    """
    # set parameters
    FIRING_RATE_UNIT = 0.01
    BANDWITH = 1e-1

    # get firing rates
    firings = np.hstack([true_firing, sorter_firing])

    # find the common firing rate support
    firing_rate_support = np.arange(
        min(firings), max(firings), FIRING_RATE_UNIT
    )

    # estimate density (probability) for ground truth
    kde_true = scipy.stats.gaussian_kde(true_firing, bw_method=BANDWITH)
    kde_true = kde_true.evaluate(firing_rate_support)
    kde_true_p = kde_true / sum(kde_true)

    # estimate density ((probability) for sorter
    kde_sorter = scipy.stats.gaussian_kde(sorter_firing, bw_method=BANDWITH)
    kde_sorter = kde_sorter.evaluate(firing_rate_support)
    kde_sorter_p = kde_sorter / sum(kde_sorter)

    # calculate bias
    return (kde_sorter_p - kde_true_p, firing_rate_support)


def get_outliers(data: pd.Series):
    """detect outliers

    Args:
        data (pd.Series): _description_

    Returns:
        _type_: _description_
    """
    q1_gt, q3_gt = np.percentile(data, [25, 75])
    whisker_low_gt = q1_gt - (q3_gt - q1_gt) * 1.5
    whisker_high_gt = q3_gt + (q3_gt - q1_gt) * 1.5
    return data[(data > whisker_high_gt) | (data < whisker_low_gt)]
    return data[(data > whisker_high_gt) | (data < whisker_low_gt)]
    return data[(data > whisker_high_gt) | (data < whisker_low_gt)]
    return data[(data > whisker_high_gt) | (data < whisker_low_gt)]
    return data[(data > whisker_high_gt) | (data < whisker_low_gt)]
    return data[(data > whisker_high_gt) | (data < whisker_low_gt)]
    return data[(data > whisker_high_gt) | (data < whisker_low_gt)]
