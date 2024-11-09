import shutil

import numpy as np
import pandas as pd
import spikeinterface.postprocessing as spost
from matplotlib import pyplot as plt
from spikeinterface.comparison import GroundTruthStudy

from src.nodes.dataeng.silico.filtering import create_study_object


def get_waveform_extractor_and_ground_truth(
    study_object: dict, study_folder: str
):
    """create SpikeInterface "WaveformExtractor" object

    Args:
        study_object (dict): _description_
        study_folder (str): _description_

    Returns:
        _type_: _description_
    """

    # overwrite existing data
    shutil.rmtree(study_folder, ignore_errors=True)
    study = GroundTruthStudy.create(study_folder, study_object)

    # get waveforms
    study.compute_waveforms(study_object["rec0"][0], ms_before=2, ms_after=2)
    waveform_extractor = study.get_waveform_extractor(study_object["rec0"][0])
    ground_truth = study.get_ground_truth(study_object["rec0"][0])
    return {
        "waveform_extractor": waveform_extractor,
        "GroundTruth": ground_truth,
    }


def plot_unit_waveform(
    waveform_extractor: dict, unit_id: np.array, **kwargs: dict
):
    """Plot each cell spike waveform

    Args:
        waveform_extractor (dict): _description_
        unit_id (np.array): _description_

    Kwargs:
        "plot_options"
            "show":
            "show_and_save":
            "save_only":
    """

    # plot
    colors = ["Olive", "Teal", "Fuchsia"]

    # get channels where spike amplitude is maximal
    max_chids = spost.get_template_extremum_channel(
        waveform_extractor, peak_sign="neg"
    )

    # plot each cell waveform extracted from its best channel
    for _, unit_id in enumerate(unit_id):

        # get cell waveforms
        waveform = waveform_extractor.get_waveforms(unit_id)

        # get best channel index in array
        idx = int(max_chids[unit_id]) - 64

        # set plot
        plt.figure(figsize=(1.9, 1.9))

        # plot waveform instances
        plt.plot(waveform[:, :, idx - 1].T, color=colors[1], lw=0.3)

        # plot average waveform
        tmp = waveform_extractor.get_template(unit_id)
        plt.plot(tmp[:, idx - 1], color=colors[2])
        plt.title(
            f"Cell id: {unit_id} on Channel {max_chids[unit_id]}", fontsize=6
        )

        # set display and save options
        if "plot_options" in kwargs:
            if kwargs["plot_options"] == "show":
                plt.show()
            elif kwargs["plot_options"] == "show_and_save":
                plt.show()
                plt.savefig(f"waveform_{unit_id}")
            elif kwargs["plot_options"] == "save_only":
                plt.savefig(f"waveform_{unit_id}")
                plt.close()


def plot_neuron_best_waveform(
    traces,
    sorting_object,
    waveform_extractor: dict,
    study_folder: str,
    **kwargs: dict,
):
    """Plot neurons best waveforms extracted from spike triggered average of the channel traces
    where the neuron has the extrema spike amplitude

    Args:
        traces (_type_): _description_
        sorting_object (_type_): _description_
        waveform_extractor (dict): _description_
        study_folder (str): _description_

    Kwargs:

    """
    # buil sorter comparison object
    study_object = create_study_object(traces, sorting_object)

    # build extractor
    waveform_extractor = viz.get_waveform_extractor_and_ground_truth(
        study_object=study_object, study_folder=study_folder
    )

    # plot waveform
    viz.plot_unit_waveform(
        waveform_extractor=waveform_extractor["waveform_extractor"],
        unit_id=waveform_extractor["GroundTruth"].unit_ids,
        **kwargs,
    )


def plot_probe_with_circuit(
    cell_coord: pd.DataFrame,
    target_cell_coord: pd.DataFrame,
    probe_coord: np.array,
    contact_ids: np.array,
):
    """probe probe contacts (gold) with mircocircuit (blue) and target cells (red)

    Args:
        cell_coord (pd.DataFrame): _description_
        target_cell_coord (pd.DataFrame): _description_
        probe_coord (np.array): _description_

    Returns:
        _type_: _description_
    """

    # Set parameters
    CONTACT_SIZE = 8
    CONTACT_COLOR = "w"
    VIEW_INITS = [[0, 70], [0, 0], [0, 90], [50, -5, 60]]
    SUBPLOTS = [151, 152, 153, 154]

    # View in contact channel locations in 3D
    fig = plt.figure(figsize=(15, 3))

    for ix in range(len(VIEW_INITS)):
        ax = fig.add_subplot(SUBPLOTS[ix], projection="3d")
        ax.view_init(*VIEW_INITS[ix])

        # plot cells
        ax.plot(
            cell_coord.x,
            cell_coord.y,
            cell_coord.z,
            ".",
            alpha=0.05,
        )

        # plot cells near contacts
        ax.plot(
            target_cell_coord.x,
            target_cell_coord.y,
            target_cell_coord.z,
            "r.",
            alpha=0.05,
        )

        # plot probe contacts
        ax.plot(
            probe_coord[:, 0],
            probe_coord[:, 1],
            probe_coord[:, 2],
            marker=".",
            markersize=CONTACT_SIZE,
            color=CONTACT_COLOR,
            markeredgecolor="black",
            linestyle="None",
        )

        # legend
        ax.set_xlim([3500, 4750])
        ax.set_zlim([-2600, -1600])
        ax.set_ylim([-1900, 0])
        # ax.tick_params(axis="both", which="major", labelsize=8)
        ax.axis("off")

    # View contact locations in 3D
    ax = fig.add_subplot(155, projection="3d")
    ax.view_init(0, 0)
    ax.plot(
        probe_coord[:, 0],
        probe_coord[:, 1],
        probe_coord[:, 2],
        ".",
        markersize=CONTACT_SIZE,
        color=CONTACT_COLOR,
        markeredgecolor="black",
        linestyle="None",
    )
    ax.set_xlim([3500, 4750])
    ax.set_zlim([-2600, -1600])
    ax.set_ylim([-1900, 0])
    # ax.tick_params(axis="both", which="major", labelsize=8)
    ax.axis("off")

    # annotate contacts with multiple of 2 ids
    # (for visibility)
    # for ix in range(len(probe_coord)):
    #     if ix % 15 == 0:
    #         ax.text(
    #             probe_coord[ix, 0],
    #             probe_coord[ix, 1] + 12,
    #             probe_coord[ix, 2],
    #             "%s" % (str(contact_ids[ix])),
    #             size=5,
    #             zorder=1,
    #             color="k",
    #         )
    plt.tight_layout()
    return fig


def plot_neuropixel_probe_with_circuit(
    cell_coord: pd.DataFrame,
    probe_coord: np.array,
):
    """probe probe contacts (gold) with mircocircuit (blue) and target cells (red)

    Args:
        cell_coord (pd.DataFrame): _description_
        target_cell_coord (pd.DataFrame): _description_
        probe_coord (np.array): _description_

    Returns:
        _type_: _description_
    """

    # View in contact channel locations in 3D
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(151, projection="3d")
    ax.view_init(10, 70)
    ax.plot(
        cell_coord.x,
        cell_coord.y,
        cell_coord.z,
        ".",
        alpha=0.05,
    )
    ax.plot(
        probe_coord[:, 0],
        probe_coord[:, 1],
        probe_coord[:, 2],
        marker=".",
        markersize=10,
        color="y",
        markeredgecolor="black",
        linestyle="None",
    )

    ax.set_xlim([3500, 4750])
    ax.set_zlim([-2600, -1600])
    ax.set_ylim([-1900, 0])

    # angle view 2
    ax = fig.add_subplot(152, projection="3d")
    ax.view_init(0, 0)
    ax.plot(
        cell_coord.x,
        cell_coord.y,
        cell_coord.z,
        ".",
        alpha=0.05,
    )
    ax.plot(
        probe_coord[:, 0],
        probe_coord[:, 1],
        probe_coord[:, 2],
        marker=".",
        markersize=10,
        color="y",
        markeredgecolor="black",
        linestyle="None",
    )
    ax.set_xlim([3500, 4750])
    ax.set_zlim([-2600, -1600])
    ax.set_ylim([-1900, 0])

    # angle view 3
    ax = fig.add_subplot(153, projection="3d")
    ax.view_init(0, 90)
    ax.plot(
        cell_coord.x,
        cell_coord.y,
        cell_coord.z,
        ".",
        alpha=0.05,
    )
    ax.plot(
        probe_coord[:, 0],
        probe_coord[:, 1],
        probe_coord[:, 2],
        marker=".",
        markersize=10,
        color="y",
        markeredgecolor="black",
        linestyle="None",
    )

    ax.set_xlim([3500, 4750])
    ax.set_zlim([-2600, -1600])
    ax.set_ylim([-1900, 0])

    # angle view 4
    ax = fig.add_subplot(154, projection="3d")
    ax.view_init(50, -5, 60)
    ax.plot(
        cell_coord.x,
        cell_coord.y,
        cell_coord.z,
        ".",
        alpha=0.05,
    )
    ax.plot(
        probe_coord[:, 0],
        probe_coord[:, 1],
        probe_coord[:, 2],
        marker=".",
        markersize=10,
        color="y",
        markeredgecolor="black",
        linestyle="None",
    )
    ax.set_xlim([3500, 4750])
    ax.set_zlim([-2600, -1600])
    ax.set_ylim([-1900, 0])
    plt.tight_layout()
    return fig
