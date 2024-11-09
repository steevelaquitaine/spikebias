"""Pipeline that plots histogram of cell sorting errors

usage:

    # activate python environment 
    source env_kilosort_silico/bin/activate

    # run pipeline as module
    python3 -m src.pipes.figures.run_histo_detected_missed_neuropixels_2023_02_19

    # ... or 
    python3 src/pipes/sorting/run_histo_sorting_errors_neuropixels_2023_02_19.py
"""

import pandas as pd
import spikeinterface as si
from matplotlib import pyplot as plt
from spikeinterface import comparison

from src.nodes.utils import get_config_silico_neuropixels

# SET RUN PARAMETERS
SIMULATION_DATE = "2023_02_19"

# GET CONFIG
data_conf, param_conf = get_config_silico_neuropixels(SIMULATION_DATE).values()

# SET GROUND TRUTH SPIKES AND CELLS PATH
# computed by match_sorted_to_true_neuropixels_2023_02_19.py
CELL_MATCHING_PATH = data_conf["postprocessing"]["cell_matching"]

# SET FIGURE PATH
FIG_PATH = data_conf["figures"]["silico"]["cell_detected_missed"]


def run(TruthExtractor_path: str, SortingExtractor_path: str):
    """plots histogram of detected vs. misses (cell sorting biases)

    Args:
        TruthExtractor_path (str): _description_
        SortingExtractor_path (str): _description_

    Usage:
        # set Kilosort sorted spikes and cells path
        KS3_SORTING_PATH = data_conf["sorting"]["sorters"]["kilosort3"]["output"]

        # set ground truth spikes and cells path
        GT_SORTING_PATH = data_conf["sorting"]["simulation"]["ground_truth"]["output"]

        # run
        from src.pipes.figures import run_histo_detected_missed_neuropixels_2023_02_19 as pipe
        out = pipe.run(GT_SORTING_PATH, KS3_SORTING_PATH)
    """
    # load Kilosort3 Sorted spikes and cells
    SortingExtractorKS3 = si.load_extractor(SortingExtractor_path)

    # load ground truth spikes and cells
    GTSortingExtractor = si.load_extractor(TruthExtractor_path)

    # agreement score between sorted and true cells
    MatchingObject = comparison.compare_sorter_to_ground_truth(
        GTSortingExtractor, SortingExtractorKS3, exhaustive_gt=True
    )
    # count detected units (agreement > 0)
    detected_true_units = MatchingObject.agreement_scores.sum(axis=1) > 0
    n_detected = sum(detected_true_units)

    # count missed units (agreement > 0)
    n_missed = sum(detected_true_units == False)

    # build histo dataset
    data_df = pd.DataFrame(
        {
            "cell count": [
                n_detected,
                n_missed,
            ],
        },
        index=["detected", "missed"],
    )

    # set figure
    fig, axis = plt.subplots(1, 1, figsize=(5, 2))

    # set plot legend
    # colors = [[0.87, 0.92, 0.96], "w"]
    colors = ["w", [0.3, 0.3, 0.3]]
    txt_colors = ["k", "w"]

    # plot
    ax = data_df.T.plot.barh(
        ax=axis,
        stacked=True,
        color=colors,
        width=0.2,
        edgecolor="k",
        linewidth=0.6,
    )

    # set axis legend
    ax.spines[["left", "right", "top"]].set_visible(False)
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(True)
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)
    ax.set_xlabel("true cells (count)", fontsize=9)
    ax.legend(
        bbox_to_anchor=(0.6, 0.6),
        frameon=False,
        fontsize=9,
        handletextpad=0.6,
    )
    ax.set_title("the sorter misses most true cells", fontsize=9)
    ax.tick_params(axis="both", which="major", labelsize=9)
    plt.tight_layout()

    # annotate bars with count by sorting error type
    for p_i, patch in enumerate(ax.patches):
        width, height = patch.get_width(), patch.get_height()
        x, y = patch.get_xy()
        ax.text(
            x + width / 2,
            y + height / 2,
            "{:.0f}".format(width),
            horizontalalignment="center",
            verticalalignment="center",
            color=txt_colors[p_i],
        )

    # write
    fig.savefig(FIG_PATH)

    # count true units
    n_true_units = len(MatchingObject.agreement_scores.index)

    # test that sorting outcome type covers all true cells
    assert (
        data_df["cell count"].sum() == n_true_units
    ), "total count over cell types does not equal true cell count"
    return {
        "fig": fig,
        "errors_count": data_df,
        "MatchingObject": MatchingObject,
    }


if __name__ == "__main__":
    run()
