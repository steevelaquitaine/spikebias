"""Pipeline that plots probability (p-)bias cell factors analysis
author: steeve.laquitaine@epfl.ch  

usage:

    # activate python environment 
    source env_kilosort_silico/bin/activate

    # run pipeline as module
    python3 -m src.pipes.figures.p_bias_factors

    # ... or 
    python3 src/pipes/figures/p_bias_factors.py
"""


# SETUP PACKAGES
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# from src.nodes.postpro.cell_matching import match_firing_rate
from src.nodes.utils import get_config
from src.pipes.figures import (
    run_histo_sorting_errors_neuropixels_2023_02_19 as run_se,
)

# SET PROJECT PATH
EXPERIMENT = "silico_neuropixels"
SIMULATION_DATE = "2023_02_19"

# SET RUN CONFIG
data_conf, _ = get_config(EXPERIMENT, SIMULATION_DATE).values()

# SET PARAMETERS
# LOW_RATE_CEILING = 0.2  # max firing rate where negative proba change is observed in "bias plot"
# MID_RATE_CEILING = (
# 1  # max firing rate where positive proba change is observed in "bias plot"
# )

# SET FIGURE PATH
FIG_PATH = data_conf["figures"]["silico"]["p_bias_factors"]


def get_missed_vs_detected(cell_matching_new: pd.DataFrame):
    # select bias explanatory features
    df = cell_matching_new.sort_values(by="true firing rate", ascending=True)

    #  set missed vs. detected feature
    missed = df[df["agreement_score"].isnull()]
    detected = df[df["agreement_score"].notnull()]
    return {"missed": missed, "detected": detected}


# def plot_missed_cells_rate_hist(ax, missed_cells: pd.DataFrame, title: str):
#     X_MAX = 4.4
#     BIN_STEP = 0.2

#     # set firing rate bins
#     BINS = np.arange(0, X_MAX, BIN_STEP)

#     # plot histogram
#     ax = missed_cells["true firing rate"].hist(
#         bins=BINS, width=0.17, color=[0.3, 0.3, 0.3]
#     )
#     ax.set_xlabel("firing rate (spikes/sec)", fontsize=9)
#     ax.set_ylabel("missed cells (count)", fontsize=9)
#     if title:
#         ax.set_title(title, fontsize=9)
#     ax.set_xticks(np.round(np.arange(0, X_MAX, 0.4), 1))
#     ax.set_xticklabels(np.round(np.arange(0, X_MAX, 0.4), 1), fontsize=9)
#     ax.grid(False)
#     ax.spines[["right", "top"]].set_visible(False)
#     ax.tick_params(axis="both", which="major", labelsize=9)
#     ax.patches[0].set_color("r")
#     return ax


def plot_missed_cells_rate_hist(
    ax,
    missed_cells: pd.DataFrame,
    x_max: float = 4.4,
    bin_step: float = 0.2,
    bin_width: float = 0.17,
    title: str = "histogram",
):
    # set firing rate bins
    BINS = np.arange(0, x_max, bin_step)

    # plot histogram
    ax = missed_cells["true firing rate"].hist(
        bins=BINS, width=bin_width, color=[0.3, 0.3, 0.3]
    )

    # legend
    ax.set_xlabel("firing rate (spikes/sec)", fontsize=9)
    ax.set_ylabel("missed cells (count)", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(False)
    ax.spines[["right", "top"]].set_visible(False)
    return ax


# def engineer_features(detected_cells: pd.DataFrame):
#     # Engineer some chosen detected cells' features
#     feature_set = [
#         "oversplit_true_cell",
#         "agreement_score",
#         "true firing rate",
#         "ks3 firing rate",
#     ]
#     features = detected_cells[feature_set]

#     # biased firing rate range feature
#     features["true_firing_rate_feat"] = features["true firing rate"]
#     features["true_firing_rate_feat"][
#         features["true firing rate"] < LOW_RATE_CEILING
#     ] = "neg_p_bias"
#     features["true_firing_rate_feat"][
#         np.logical_and(
#             features["true firing rate"] >= LOW_RATE_CEILING,
#             features["true firing rate"] < MID_RATE_CEILING,
#         )
#     ] = "pos_p_bias"
#     features["true_firing_rate_feat"][
#         features["true firing rate"] >= MID_RATE_CEILING
#     ] = "no_p_bias"

#     # "rate change" feature
#     features["rate_change"] = (
#         features["ks3 firing rate"] - features["true firing rate"]
#     )
#     features["rate_change_feat"] = features["rate_change"]
#     features["rate_change_feat"][features["rate_change"] > 0] = "increases"
#     features["rate_change_feat"][features["rate_change"] < 0] = "decreases"
#     features["rate_change_feat"][features["rate_change"] == 0] = "unchanged"
#     features

#     # "oversplit" feature
#     features["oversplit_feat"] = features["oversplit_true_cell"]
#     features["oversplit_feat"][
#         features["oversplit_true_cell"] == True
#     ] = "oversplit"
#     features["oversplit_feat"][
#         features["oversplit_true_cell"] == False
#     ] = "unique"
#     features.sort_values(by=["true firing rate"])
#     return features


def engineer_detected_unit_features(
    detected_cells: pd.DataFrame,
    LOW_RATE_CEILING: float = 0.2,
    MID_RATE_CEILING: float = 1,
):
    # select features
    feature_set = [
        "oversplit",
        "true firing rate",
        "ks3 firing rate",
    ]
    feats = detected_cells[feature_set]
    true_rate = feats["true firing rate"]

    # get data for units with Firing rate below LOW_RATE_CEILING
    # default 0.2 Hz or "neg_p_bias" units
    feats["true_firing_rate_feat"] = true_rate
    neg_p_bias_loc = true_rate < LOW_RATE_CEILING
    feats["true_firing_rate_feat"][neg_p_bias_loc] = "neg_p_bias"

    # get data for units with Firing rate in between LOW_RATE_CEILING
    # and MID_RATE_CEILING (default 1 Hz) or "pos_p_bias" units
    pos_p_bias_loc = np.logical_and(
        true_rate >= LOW_RATE_CEILING,
        true_rate < MID_RATE_CEILING,
    )
    feats["true_firing_rate_feat"][pos_p_bias_loc] = "pos_p_bias"

    # get data for units with firing rate above MID_RATE_CEILING
    # or "no_p_bias" units
    feats["true_firing_rate_feat"][true_rate >= MID_RATE_CEILING] = "no_p_bias"

    # create "rate change" feature
    feats["rate_change"] = feats["ks3 firing rate"] - true_rate
    feats["rate_change_feat"] = feats["rate_change"]
    feats["rate_change_feat"][feats["rate_change"] > 0] = "increases"
    feats["rate_change_feat"][feats["rate_change"] < 0] = "decreases"
    feats["rate_change_feat"][feats["rate_change"] == 0] = "unchanged"

    # create "oversplit" feature
    feats["oversplit_feat"] = feats["oversplit"]
    feats["oversplit_feat"][feats["oversplit"] == True] = "oversplit"
    feats["oversplit_feat"][feats["oversplit"] == False] = "unique"
    feats.sort_values(by=["true firing rate"])
    return feats


def get_drivers_of_density_biases(
    df_detected_features: pd.DataFrame,
    LOW_RATE_CEILING: float = 0.2,
    MID_RATE_CEILING: float = 1,
):
    # get all overestimated true units firing below the
    # LOW_RATE_CEILING = 0.2Hz (default) - that were matched with
    # a sorting unit that fired above 0.2 Hz
    overest_sparse_units = df_detected_features[
        (df_detected_features["true_firing_rate_feat"] == "neg_p_bias")
        & (df_detected_features["ks3 firing rate"] > LOW_RATE_CEILING)
    ]

    # find mid. firing cells that move to a lower firing rate range, reducing the negative density bias
    # get all underestimated true units firing above LOW_RATE_CEILING=0.2Hz
    # and that were matched with sorted units that fired below 0.2 Hz
    underest_mid_rate = df_detected_features[
        (df_detected_features["true_firing_rate_feat"] == "pos_p_bias")
        & (df_detected_features["ks3 firing rate"] < LOW_RATE_CEILING)
    ]

    # find mid. firing cells that move to a higher firing rate range, biasing higher firing density
    overest_mid_rate = df_detected_features[
        (df_detected_features["true_firing_rate_feat"] == "pos_p_bias")
        & (df_detected_features["ks3 firing rate"] > MID_RATE_CEILING)
    ]

    biased_cells = pd.concat(
        [
            overest_sparse_units,
            underest_mid_rate,
            overest_mid_rate,
        ]
    )

    # Get all cells not causing bias
    # ------------------------------
    # find low firing cells that stay in their firing rate range (do not contribute to bias)
    unbiased_sparse_units = df_detected_features[
        (df_detected_features["true_firing_rate_feat"] == "neg_p_bias")
        & (df_detected_features["ks3 firing rate"] < LOW_RATE_CEILING)
    ]

    # find mid firing cells that stay in their range, causing no bias
    unbiased_mid_rate_cells = df_detected_features[
        (df_detected_features["true_firing_rate_feat"] == "pos_p_bias")
        & (df_detected_features["ks3 firing rate"] > LOW_RATE_CEILING)
        & (df_detected_features["ks3 firing rate"] < MID_RATE_CEILING)
    ]

    # plot cells contributing not apparent bias
    unbiased_high_rate_cells = df_detected_features[
        df_detected_features["true_firing_rate_feat"] == "no_p_bias"
    ]

    unbiased_cells = pd.concat(
        [
            unbiased_sparse_units,
            unbiased_mid_rate_cells,
            unbiased_high_rate_cells,
        ]
    )
    return {
        "biased_cells": biased_cells,
        "unbiased_cells": unbiased_cells,
        "overest_sparse_units": overest_sparse_units,
        "underest_mid_rate": underest_mid_rate,
        "overest_mid_rate": overest_mid_rate,
    }


def plot_cell_drivers_of_bias(
    ax,
    overest_sparse_units: pd.DataFrame,
    up_biased_mid_rate_cells: pd.DataFrame,
    down_biased_mid_rate_cells: pd.DataFrame,
    unbiased_cells: pd.DataFrame,
    x_max: float,
    y_max: float,
    legend: bool,
    title: bool,
    ylabel: bool,
):
    DOT_SIZE = 6
    DOT_LWIDTH = 0.1

    # plot
    overest_sparse_units.plot(
        ax=ax,
        x="true firing rate",
        y="ks3 firing rate",
        marker="o",
        linestyle="None",
        markeredgecolor="w",
        color=[1, 0, 0],
        markersize=DOT_SIZE,
        linewidth=DOT_LWIDTH,
        label="overest. sparse units",
    )

    # plot
    up_biased_mid_rate_cells.plot(
        ax=ax,
        x="true firing rate",
        y="ks3 firing rate",
        marker="o",
        linestyle="None",
        markeredgecolor="w",
        color=[0.3, 0.3, 1],
        markersize=DOT_SIZE,
        linewidth=DOT_LWIDTH,
        label="underest. mid-rate",
    )
    # plot
    down_biased_mid_rate_cells.plot(
        ax=ax,
        x="true firing rate",
        y="ks3 firing rate",
        marker="o",
        linestyle="None",
        markeredgecolor="w",
        color=[0.6, 0.6, 1],
        markersize=DOT_SIZE,
        linewidth=DOT_LWIDTH,
        label="overest. mid-rate",
    )

    # plot
    unbiased_cells.plot(
        ax=ax,
        x="true firing rate",
        y="ks3 firing rate",
        marker="o",
        linestyle="None",
        markeredgecolor=[0.1, 0.1, 0.1],
        markerfacecolor="none",
        markersize=DOT_SIZE,
        linewidth=DOT_LWIDTH,
        label="other cells",
        alpha=0.2,
    )

    # plot diagonal
    ax.plot(
        [0, x_max], [0, y_max], color=[0.3, 0.3, 0.3], linestyle=":", zorder=-1
    )

    # legend
    ax.set_xlim([-0.1, x_max])
    ax.set_ylim([0, y_max])
    ax.spines[["right", "top"]].set_visible(False)
    if legend:
        ax.legend(
            loc="upper right", frameon=False, fontsize=9, handletextpad=0.007
        )
    else:
        ax.get_legend().remove()
    if ylabel:
        ax.set_ylabel(
            r"detected cells' firing rate" "\n" "(spikes/sec)", fontsize=9
        )
    ax.set_xlabel("true cells' firing rate (spike/sec)", fontsize=9)
    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    return ax


def run():
    # get match between true and sorted
    out = run_se.run()
    cell_matching = out["cell_matching"]

    # match true and sorted cells firing rates
    cell_matching = match_firing_rate(cell_matching, data_conf)

    # get missed vs. detected cells
    missed_cells, detected_cells = get_missed_vs_detected(
        cell_matching
    ).values()

    # plot missed cells firing rate histogram
    fig = plt.figure(figsize=(5, 5))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, :])
    ax1 = plot_missed_cells_rate_hist(
        ax1, missed_cells, "cell misses drive negative p-bias"
    )

    # get cell drivers of density biases
    features = engineer_features(detected_cells)
    out = get_drivers_of_density_biases(
        features, LOW_RATE_CEILING, MID_RATE_CEILING
    )

    # plot
    ax2 = fig.add_subplot(gs[1, 0])
    ax2 = plot_cell_drivers_of_bias(
        ax2,
        out["biased_cells"],
        out["unbiased_cells"],
        9,
        9,
        legend=True,
        title=r"low-FR cells" "\n" "are main driver of p-biases",
        ylabel=True,
    )

    # plot zoom-in
    ax3 = fig.add_subplot(gs[1, 1])
    ax3 = plot_cell_drivers_of_bias(
        ax3,
        out["biased_cells"],
        out["unbiased_cells"],
        3.5,
        3.5,
        legend=False,
        title="zoom in",
        ylabel=False,
    )
    plt.tight_layout()

    # write
    fig.savefig(FIG_PATH)


if __name__ == "__main__":
    run()
