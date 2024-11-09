import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def tag_missed_cells(cell_matching: pd.DataFrame):
    """get the missed cell dataset

    Args:
        cell_matching (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    cell_matching["missed_cells"] = np.where(
        cell_matching["agreement_score"].isnull(), True, False
    )
    return cell_matching


# def get_missed_vs_detected(cell_matching_new: pd.DataFrame):
#     """_summary_

#     Args:
#         cell_matching_new (pd.DataFrame): _description_

#     Returns:
#         _type_: _description_
#     """
#     # select bias explanatory features
#     df = cell_matching_new.sort_values(by="true firing rate", ascending=True)

#     #  set missed vs. detected feature
#     missed = df[df["agreement_score"].isnull()]
#     detected = df[df["agreement_score"].notnull()]
#     return {"missed": missed, "detected": detected}


def plot_missed_cells_rate_hist(
    ax, missed_cells: pd.DataFrame, title: str, **vararg: dict
):
    """_summary_

    Args:
        ax (_type_): _description_
        missed_cells (pd.DataFrame): _description_
        title (str): _description_

    vararg:
        bin_set: firing rate bin size. e.g., bin_step=0.2 spikes/s
        max_rate: histogram x-axis' firing rate
        x_step: histogram x-axis' x ticklabel unit
        logscale:plot y-axis as log

    Returns:
        _type_: axis
    """

    # set default parameters
    MAX_RATE = 4.4
    BIN_STEP = 0.2
    X_STEP = 0.4

    # get input parameters, if set
    if "bin_step" in vararg:
        BIN_STEP = vararg["bin_step"]
    if "max_rate" in vararg:
        MAX_RATE = vararg["max_rate"]
    if "x_step" in vararg:
        X_STEP = vararg["x_step"]
    if "logscale" in vararg:
        ax.set_yscale("log")

    # set firing rate bins
    BINS = np.arange(0, MAX_RATE, BIN_STEP)

    # plot histogram
    ax = missed_cells["true firing rate"].hist(
        ax=ax, bins=BINS, width=0.17, color=[0.3, 0.3, 0.3]
    )

    # legend
    ax.set_xlabel("firing rate of missed cells (spikes/sec)", fontsize=9)
    ax.set_ylabel("count", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.set_xticks(np.round(np.arange(0, MAX_RATE, X_STEP), 1))
    ax.set_xticklabels(np.round(np.arange(0, MAX_RATE, X_STEP), 1), fontsize=9)
    ax.grid(False)
    ax.spines[["right", "top"]].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=9)
