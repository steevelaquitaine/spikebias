"""Pipeline that plots histogram of cell sorting errors
author: steeve.laquitaine@epfl.ch

usage:

    # activate python environment 
    source env_kilosort_silico/bin/activate

    # run pipeline as module
    python3 -m src.pipes.sorting.run_histo_sorting_errors_neuropixels_2023_02_19

    # ... or 
    python3 src/pipes/sorting/run_histo_sorting_errors_neuropixels_2023_02_19.py
"""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.nodes.utils import get_config_silico_neuropixels as get_conf

SIMULATION_DATE = "2023_02_19"

# GET CONFIG
data_conf, param_conf = get_conf(SIMULATION_DATE).values()

# SET GROUND TRUTH SPIKES AND CELLS PATH
# computed by match_sorted_to_true_neuropixels_2023_02_19.py
CELL_MATCHING_PATH = data_conf["postprocessing"]["cell_matching"]

# SET FIGURE PATH
FIG_PATH = data_conf["figures"]["silico"]["cell_sorting_errors"]


def run(**vararg: dict):
    """plot histogram of cell-level sorting errors

    Returns:
        _type_: _description_
    """
    # read cell matching dataset
    cell_matching = pd.read_parquet(CELL_MATCHING_PATH)

    # count true cells
    n_true_cells = cell_matching["true_cell_match"].nunique()

    # get well detected cells
    well_detected_true_cells = cell_matching[
        cell_matching["oversplit_true_cell"] == False
    ]
    n_well_detected_true_cells = len(well_detected_true_cells)

    # count oversplit cells
    oversplit_true_cells = cell_matching[
        cell_matching["oversplit_true_cell"] == True
    ]
    n_oversplit_true_cells = len(
        oversplit_true_cells["true_cell_match"].unique()
    )

    # count missed true cells
    missed_true_cells = cell_matching["true_cell_match"][
        np.logical_and(
            cell_matching["true_cell_match"].notna(),
            cell_matching["sorted_cell"].isna(),
        )
    ]
    n_missed_true_cells = len(missed_true_cells)

    # build dataset
    data_df = pd.DataFrame(
        {
            "cell count": [
                n_well_detected_true_cells,
                n_oversplit_true_cells,
                n_missed_true_cells,
            ],
        },
        index=["well detected", "oversplit", "missed"],
    )

    # case plot
    if not "no_plot" in vararg:
        # set figure
        fig, axis = plt.subplots(1, 1, figsize=(5, 2))

        # set plot legend
        colors = [[0.87, 0.92, 0.96], [0.61, 0.79, 0.88], "w"]
        txt_colors = ["k", "k", "k"]

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
        ax.set_xlabel("true cell (count)")
        ax.legend(bbox_to_anchor=(0.7, 0.7), frameon=False)
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

        fig.savefig(FIG_PATH)

    # test that sorting outcome type covers all true cells
    assert (
        data_df["cell count"].sum() == n_true_cells
    ), "total count over cell types does not equal true cell count"

    return {"errors_count": data_df, "cell_matching": cell_matching}


if __name__ == "__main__":
    run()
