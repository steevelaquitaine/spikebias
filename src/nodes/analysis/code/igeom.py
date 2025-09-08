"""node that calculates information geometrics (capacity, etc..)
from the biophysical simulation of the neuropixels
evoked regime

author: steeve.laquitaine@epfl.ch

Usage: 
    
    pipelines:
        * sbatch cluster/analysis/npx_evoked/full/code/igeom.sh
        * python3.9 -m src.pipes.analysis.npx_evoked.full.code.by_sampling.igeom
    
Returns:
    _type_: _description_

Yields:
    _type_: _description_
"""
import os
import pickle
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from mftma.manifold_analysis_correlation import manifold_analysis_corr
import random
import spikeinterface as si 
from collections import defaultdict
import time
from spikeinterface import comparison
import yaml
import logging
import logging.config
from src.nodes.analysis.features import features as feat

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# Stimulus discrimination task ------------------------------

def get_task_parameters(
    start_delay=500,
    n_orientations=10,
    n_repeats=50,
    stimulus_duration=200,
    n_simulations=36,
):
    """_summary_

    Args:
        start_delay (int, optional): _description_. Defaults to 500.
        - when the first stimulus is applied
        n_orientations (int, optional): _description_. Defaults to 10.
        n_repeats (int, optional): _description_. Defaults to 50.
        stimulus_duration (int, optional): _description_. Defaults to 200.
        n_simulations (int, optional): _description_. Defaults to 36.
        - there are 36 simulation files

    Returns:
        _type_: _description_
    """
    # get epoch timings
    epoch_ms = n_simulations * (
        [start_delay] + n_orientations * n_repeats * [stimulus_duration]
    )
    # get epoch labels
    epoch_labels = n_simulations * (
        ["delay"] + n_orientations * n_repeats * ["stimulus"]
    )

    return {"epoch_labels": epoch_labels, "epoch_ms": epoch_ms}


def get_stimulus_intervals_ms(epoch_labels, epoch_ms):

    # find stimulus epoch starts and ends
    epoch_end_ms = np.cumsum(epoch_ms)
    epoch_start_ms = np.hstack([0, epoch_end_ms])[:-1]
    df = pd.DataFrame(data=epoch_end_ms)
    df.columns = ["end"]
    df.insert(0, "start", epoch_start_ms)
    df.index = epoch_labels

    # get stimulus intervals
    return [tuple(df.iloc[ix]) for ix in range(len(df)) if df.index[ix] == "stimulus"]


def get_stimulus_labels():
    start = np.arange(0, 360, 10)
    end = np.arange(10, 370, 10)

    stimulus_labels = []
    for ix in range(36):
        stimulus_labels.append(np.repeat(np.arange(start[ix], end[ix], 1), 50))
    return np.array(stimulus_labels).flatten()


# Datasets -------------------------


def reshape_responses(responses: np.array, stimulus_labels, n_exple_per_class: int):
    """format unit responses into a list of N_CLASSES arrays of
    size N units x N_EXPLE_PER_CLASS

    Returns:
        (list[arrays]): _description_
    - e.g., dataset = [np.random.randn(5000, 50) for i in range(100)]
    """
    dataset = [
        responses[:, stimulus_labels == orientation]
        for orientation in np.unique(stimulus_labels)
    ]
    assert dataset[1].shape[1] == n_exple_per_class, "wrong shape"
    return dataset


def convert_spike_trains_to_ms(spike_trains: np.array, sfreq: int):
    """_summary_

    Args:
        spike_trains (np.array): _description_
        sfreq (int): _description_

    Returns:
        _type_: _description_
    """
    sample_ms = 1 / (sfreq / 1000)
    spike_trains_ms = spike_trains * sample_ms
    return spike_trains_ms


def get_evoked_responses(spike_trains_ms: np.array, stim_intervals_ms: np.array):
    """_summary_

    Args:
        spike_trains_ms (np.array): _description_
        stim_intervals_ms (np.array): array of tuples (start, end)

    Returns:
        _type_: _description_
    """
    # Use numpy's digitize function to find the bin indices for each value
    bins = [interval[1] for interval in stim_intervals_ms]

    # Return the indices of the bins to which each value in spike_trains_ms belongs.
    bin_indices = np.digitize(spike_trains_ms, bins=bins)

    # Use Counter to count occurrences of bin indices
    interval_counter = Counter(bin_indices)

    # find active stimulus epochs
    active_bin_spike_count = [items[1] for items in list(interval_counter.items())]
    active_bin_ix = [items[0] for items in list(interval_counter.items())]

    # cast unit responses by stimulus epoch in an array
    unit_responses = np.zeros(len(bins))
    unit_responses[active_bin_ix] = active_bin_spike_count
    return unit_responses


def compute_response_by_stim_matrix(unit_ids, Sorting, stimulus_intervals_ms):
    """compute unit response matrix (N unit response x M directions)

    Args:
        Sorting (_type_): _description_
        stimulus_intervals_ms (_type_): _description_

    Returns:
        _type_: _description_
    """
    SFREQ = Sorting.get_sampling_frequency()

    # takes 18 secs
    responses = []
    for unit_id in unit_ids:
        spike_trains = Sorting.get_unit_spike_train(unit_id)
        spike_trains_ms = convert_spike_trains_to_ms(spike_trains, SFREQ)
        responses.append(get_evoked_responses(spike_trains_ms, stimulus_intervals_ms))
    return np.array(responses)


def get_data_infogeometry_for_sorter(
    responses_sorted,
    stimulus_labels,
    sample_classes,
    n_exple_per_class,
    n_new,
    seed=0,
):
    """_summary_
    Takes 6 min for 20 classes with 50 sample per class

    Returns:
        tuple: _description_
    """
    # set seed for reproducibility
    np.random.seed(seed)

    # find actual labels indices
    classes_ix = np.array(
        [
            np.where(stimulus_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(stimulus_labels == ix)
        ]
    ).flatten()

    # analyse manifold information geometry
    data = analyse_class_manifolds(
        responses_sorted[:, classes_ix],
        stimulus_labels[classes_ix],
        n_exple_per_class,
        n_new,
    )
    return {
        "data": data,
    }
    

def get_all_sorted_responses(Sorting, stimulus_intervals_ms):
    responses = compute_response_by_stim_matrix(
        Sorting.unit_ids, Sorting, stimulus_intervals_ms
    )
    return {
        "responses": responses,
        "unit_ids": Sorting.unit_ids,
    }


def get_single_unit_responses(Sorting, stimulus_intervals_ms):
    
    # compute response matrix (unit x stimulus)
    # sorted single-units
    unit_ids = Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]

    # compute responses
    responses = compute_response_by_stim_matrix(
        unit_ids, Sorting, stimulus_intervals_ms
    )
    return {
        "responses": responses,
        "unit_ids": unit_ids,
    }
    
    
def get_unit_responses_for(unit_ids, Sorting, stimulus_intervals_ms):
    
    # compute responses
    responses = compute_response_by_stim_matrix(
        unit_ids, Sorting, stimulus_intervals_ms
    )
    return responses
    
        
def get_multiunit_sorted_responses(Sorting, stimulus_intervals_ms):
    # compute response matrix (unit x stimulus)
    # sorted multi-unit exclusively
    unit_ids = Sorting.unit_ids[Sorting.get_property("KSLabel") == "mua"]
    responses = compute_response_by_stim_matrix(
        unit_ids, Sorting, stimulus_intervals_ms
    )
    return {
        "responses": responses,
        "unit_ids": unit_ids,
    }


# Plot nodes ------------------------------

def plot_all_sorted_unit_manifolds(
    Sorting, stimulus_intervals_ms, params, params_pca, save, filename, axis_lim
):
    """plot manifolds for Kilosort and Herdingspikes
    They do not have a postprocessing operation that curates
    single units. They yield all the sorted units found.

    Args:
        Sorting (_type_): _description_
        stimulus_intervals_ms (_type_): _description_
        params (_type_): _description_
        params_pca (_type_): _description_
        save (_type_): _description_
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    FIG_SIZE = (2.8, 2.8)

    # compute response matrix (unit x stimulus)
    responses_all_sorted = compute_response_by_stim_matrix(
        Sorting.unit_ids, Sorting, stimulus_intervals_ms
    )

    # plot
    fig = plt.figure(figsize=FIG_SIZE)

    # plot 3D manifold
    ax = fig.add_subplot(2, 1, 1, projection="3d")
    manifold_sorted = plot_manifold_from_pca(
        responses_all_sorted, params_pca, ax, axis_lim
    )

    # project onto neural latents 1 and 2
    ax = fig.add_subplot(2, 1, 2)
    ax = plot_manifold_dims_1_2(manifold_sorted, params, ax, axis_lim)

    fig.subplots_adjust(wspace=0.25, hspace=0.2)

    # save
    if save:
        plt.savefig(
            filename,
            **{"transparent": True, "dpi": 300},
        )
    return (
        responses_all_sorted,
        Sorting.unit_ids,
    )


def plot_all_sorted_unit_manifolds2(
    ax, Sorting, stimulus_intervals_ms, params_pca, axis_lim
):
    """plot manifolds for Kilosort and Herdingspikes
    They do not have a postprocessing operation that curates
    single units. They yield all the sorted units found.

    Args:
        Sorting (_type_): _description_
        stimulus_intervals_ms (_type_): _description_
        params (_type_): _description_
        params_pca (_type_): _description_
        save (_type_): _description_
        filename (_type_): _description_

    Returns:
        _type_: _description_
    """
    # compute response matrix (unit x stimulus)
    responses = compute_response_by_stim_matrix(
        Sorting.unit_ids, Sorting, stimulus_intervals_ms
    )

    # plot 3D manifold
    _ = plot_manifold_from_pca(responses, params_pca, ax, axis_lim)
    return {
        "responses": responses,
        "unit_ids": Sorting.unit_ids,
    }


def plot_sorted_manifolds(
    Sorting, stimulus_intervals_ms, params, params_pca, save, filename, axis_lim
):
    # Sorted single units ------
    FIG_SIZE = (2.8, 2.8)

    # compute response matrix (unit x stimulus)
    sorted_single_unit_ids = Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]
    responses_sorted = compute_response_by_stim_matrix(
        sorted_single_unit_ids, Sorting, stimulus_intervals_ms
    )

    # plot
    fig = plt.figure(figsize=FIG_SIZE)

    # plot 3D manifold
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    manifold_sorted = plot_manifold_from_pca(responses_sorted, params_pca, ax, axis_lim)

    # project onto neural latents 1 and 2
    ax = fig.add_subplot(2, 2, 3)
    ax = plot_manifold_dims_1_2(manifold_sorted, params, ax, axis_lim)

    # All sorted ------

    # compute response matrix (unit x stimulus)
    responses_all_sorted = compute_response_by_stim_matrix(
        Sorting.unit_ids, Sorting, stimulus_intervals_ms
    )

    # plot 3D manifold
    ax = fig.add_subplot(2, 2, 2, projection="3d")
    manifold_sorted = plot_manifold_from_pca(
        responses_all_sorted, params_pca, ax, axis_lim
    )

    # project onto neural latents 1 and 2
    ax = fig.add_subplot(2, 2, 4)
    ax = plot_manifold_dims_1_2(manifold_sorted, params, ax, axis_lim)

    # fig.subplots_adjust(wspace=1, hspace=1)
    # tidy up
    fig.subplots_adjust(wspace=0.25, hspace=0.2)

    # save
    if save:
        plt.savefig(
            filename,
            **{"transparent": True, "dpi": 300},
        )
    return (
        responses_sorted,
        responses_all_sorted,
        sorted_single_unit_ids,
        Sorting.unit_ids,
    )


def plot_sorted_manifolds2(ax, Sorting, stimulus_intervals_ms, params_pca, axis_lim):
    # compute response matrix (unit x stimulus)
    # sorted single-units
    unit_ids = Sorting.unit_ids[Sorting.get_property("KSLabel") == "good"]

    # compute responses
    responses = compute_response_by_stim_matrix(
        unit_ids, Sorting, stimulus_intervals_ms
    )

    # plot 3D manifold
    _ = plot_manifold_from_pca(responses, params_pca, ax, axis_lim)
    return {
        "responses": responses,
        "unit_ids": unit_ids,
    }


def plot_manifold_from_pca(responses: np.ndarray, params: dict, ax, axis_lim, 
                           markersize=10, downsample=10):
    """PLot manifold

    Args:
      responses (np.ndarray): neurons x stimulus
    """
    # fit pca to neural responses
    pca = PCA(n_components=params["dims"])  # parametrize pca
    manifold = pca.fit_transform(responses.T)  # apply pca
    manifold = manifold / np.max(np.abs(manifold))  # normalise the values

    # setup plot parameters
    downsample = 10

    # setup plot
    plt.set_cmap("hsv")  # circular cmap

    # 3D projection
    #ax.view_init(20, 45, 0)  # elevation, azimuth, roll
    ax.view_init(0, 0, 0)  # elevation, azimuth, roll

    # color the stimulus orientations
    cmap = params["orientations"][::downsample]

    # plot neural manifold
    scat = ax.scatter(
        manifold[::downsample, 0],
        manifold[::downsample, 1],
        manifold[::downsample, 2],
        c=cmap,
        edgecolors="w",
        linewidths=0.2,
        s=markersize,        
        rasterized=True,  # memory efficiency        
    )

    # add legend
    ax.set_xlim(axis_lim)
    ax.set_ylim(axis_lim)
    ax.set_zlim(axis_lim)
    cbar = plt.colorbar(scat, ticks=[0, 90, 180, 270, 359], fraction=0.026, pad=0.04)
    cbar.ax.set_ylabel("Stimulus orientations (deg)", rotation=270, labelpad=7)
    cbar.ax.set_yticklabels([0, 90, 180, 270, 359])
    ax.set_xlabel("Neural latent 1")
    ax.set_ylabel("Neural latent 2")
    ax.set_zlabel("Neural latent 3")
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_yticks([-0.5, 0, 0.5, 1])
    ax.set_zticks([-0.5, 0, 0.5, 1])
    ax.set_xticklabels(
        [-0.5, 0, 0.5, 1], verticalalignment="baseline", horizontalalignment="right"
    )
    ax.set_yticklabels(
        [-0.5, 0, 0.5, 1], verticalalignment="baseline", horizontalalignment="right"
    )
    ax.set_zticklabels(
        [-0.5, 0, 0.5, 1], verticalalignment="baseline", horizontalalignment="right"
    )
    ax.tick_params(axis="x", which="major", pad=-3)
    ax.tick_params(axis="y", which="major", pad=-3)
    ax.tick_params(axis="z", which="major", pad=-3)
    ax.set_xlabel("Neural latent 1", labelpad=-10)
    ax.set_ylabel("Neural latent 2", labelpad=-10)
    ax.set_zlabel("Neural latent 3", labelpad=-5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.set_box_aspect((1, 1, 1))
    print("manifold axis max:", max(manifold.max(axis=1)))
    print("manifold axis min:", min(manifold.min(axis=1)))
    return {"manifold": manifold, "ax": ax}


def plot_manifold_dims_1_2(manifold: dict, params, ax, axis_lim):

    # downsample manifold for speed
    # and memory efficiency
    DOWNSAMPLE = 10

    # color the stimulus orientations
    cmap = params["orientations"][::DOWNSAMPLE]

    # plot neural manifold
    _ = ax.scatter(
        manifold["manifold"][::DOWNSAMPLE, 0],
        manifold["manifold"][::DOWNSAMPLE, 1],
        c=cmap,
        edgecolors="w",
        linewidths=0.2,
        rasterized=True,  # memory efficiency
    )

    # add legend
    ax.set_xlim(axis_lim)
    ax.set_ylim(axis_lim)
    ax.set_xticklabels([-0.5, 0, 0.5, 1])
    ax.set_yticklabels([-0.5, 0, 0.5, 1])
    ax.set_xticks([-0.5, 0, 0.5, 1])
    ax.set_yticks([-0.5, 0, 0.5, 1])
    # cbar = plt.colorbar(scat, ticks=[0, 90, 180, 270, 359], fraction=0.026, pad=0.04)
    # cbar.ax.set_ylabel("Stimulus orientations (deg)", rotation=270, labelpad=7)
    # cbar.ax.set_yticklabels([0, 90, 180, 270, 359])

    # legend
    ax.set_xlabel("Neural latent 1")
    ax.set_ylabel("Neural latent 2")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_box_aspect((1))
    return ax


# Information geometry -------------------------

def analyse_class_manifolds(responses, stimulus_labels, n_exple_per_class, n_new: int, reduce_dim=True, seed_dim_red=0):
    """_summary_

    Args:
        responses (_type_): _description_
        stimulus_labels (_type_): _description_
        n_exple_per_class (_type_): _description_
        n_new (int): number of reduced dimensions to enforce
        the same number of dimensions for all manifolds
        as the number of units can differ. This prevents confounds
        due to unequal sample size.

    Returns:
        dict: _description_
    """

    # takes 33m secs (for 20 classes)
    # - the analysis is done at zero margin (kappa)
    np.random.seed(seed_dim_red)
    kappa = 0  # margin
    n_t = 200  # number of samples

    # reshape responses
    reshaped_responses = reshape_responses(
        responses, stimulus_labels, n_exple_per_class
    )

    # Gaussian Random projection to reduce the dimensionality 
    # to the dimensionality of the unit population with the lowest
    # dimensionality that saves time and normalizes the datasets, 
    # as they have different number of units. This step should bit 
    # change the geometry too much (see the Johnson–Lindenstrauss lemma).
    projected_responses = []

    # reduce the dimensionality of each direction manifold
    # each direction manifold is for a direction
    # samples are repeats of the direction
    # n_class_manifolds x n_samples
    if reduce_dim:
        for data in reshaped_responses:
            N = data.shape[0]
            M = np.random.randn(n_new, N)
            M /= np.sqrt(np.sum(M * M, axis=1, keepdims=True))
            X = np.matmul(M, data)
            projected_responses.append(X)
        logger.info("Done with dimensionality reduction before manifold analysis...")
    else:
        projected_responses = reshaped_responses
        logger.info("No dimensionality reduction was applied before manifold analysis...")
    
    # analyse manifold geometry
    # notes
    t0 = time.time()
    capacities, radii, dimensions, corr, K = manifold_analysis_corr(
        projected_responses, kappa, n_t
    )
    print("Optimization took", np.round(time.time()-t0,2), "secs")
    return {
        "metrics_per_class": {
            "capacity": capacities,
            "radius": radii,
            "dimension": dimensions,
            "correlation": corr,
            "K": K,
        },
        "average_metrics": {
            "capacity": 1 / np.mean(1 / capacities),
            "radius": np.mean(radii),
            "dimensions": np.mean(dimensions),
            "correlation": np.mean(corr),
            "K": np.mean(K),
        },
    }


def get_true_infogeometry(
    responses,
    stimulus_labels,
    sample_classes,
    n_exple_per_class,
    n_new,
    seed_shuffling=0,
    seed_dim_red=0,
    reduce_dim=True,
):
    """_summary_
    Takes 6 min for 20 classes with 50 sample per class

    Returns:
        tuple: _description_
    """
    # set seed for reproducibility
    # use random.seed not np.random.seed
    # with random.sample
    random.seed(seed_shuffling)

    # find actual labels indices
    classes_ix = np.array(
        [
            np.where(stimulus_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(stimulus_labels == ix)
        ]
    ).flatten()

    # find shuffled labels indices to calculate
    # lower bound
    shuffled_labels = np.array(
        random.sample(
            stimulus_labels.tolist(),
            len(stimulus_labels),
        )
    )

    # find classes indices
    shuffle_classes_ix = np.array(
        [
            np.where(shuffled_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(shuffled_labels == ix)
        ]
    ).flatten()

    # analyse manifold information geometry
    data = analyse_class_manifolds(
        responses[:, classes_ix], stimulus_labels[classes_ix],
        n_exple_per_class, n_new, reduce_dim=reduce_dim,
        seed_dim_red=seed_dim_red
    )

    # get lower bound from unstructured manifolds (to normalize capacity)
    shuffled = analyse_class_manifolds(
        responses[:, shuffle_classes_ix],
        shuffled_labels[shuffle_classes_ix],
        n_exple_per_class,
        n_new,
        reduce_dim=reduce_dim, 
        seed_dim_red=seed_dim_red
    )
    return {"data": data, "shuffled": shuffled}


def get_infogeometry_for_sorter(
    responses_sorted,
    stimulus_labels,
    sample_classes,
    n_exple_per_class,
    n_new,
    seed_shuffling=0,
    seed_dim_red=0,
    reduce_dim=True
):
    """calculate information geometrical metrics
    from the responses 
    
    Takes 6 min for 20 classes with 50 sample per class

    Returns:
        tuple: _description_
    """
    # set seed for reproducibility
    # used random.seed() for random.sample() function
    # not np.random.seed()
    random.seed(seed_shuffling)

    # find actual labels indices
    classes_ix = np.array(
        [
            np.where(stimulus_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(stimulus_labels == ix)
        ]
    ).flatten()

    # find shuffled labels indices to calculate
    # lower bound
    shuffled_labels = np.array(
        random.sample(
            stimulus_labels.tolist(),
            len(stimulus_labels),
        )
    )

    # find classes indices
    shuffle_classes_ix = np.array(
        [
            np.where(shuffled_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(shuffled_labels == ix)
        ]
    ).flatten()

    # analyse manifold information geometry
    data = analyse_class_manifolds(
        responses_sorted[:, classes_ix],
        stimulus_labels[classes_ix],
        n_exple_per_class,
        n_new,
        reduce_dim=reduce_dim,
        seed_dim_red=seed_dim_red
    )

    # get lower bound from unstructed manifolds
    # (to normalize capacity)
    shuffled = analyse_class_manifolds(
        responses_sorted[:, shuffle_classes_ix],
        shuffled_labels[shuffle_classes_ix],
        n_exple_per_class,
        n_new,
        reduce_dim=reduce_dim,
        seed_dim_red=seed_dim_red
    )
    return {
        "data": data,
        "shuffled": shuffled,
    }


def get_shuffled_infogeometry_for_sorter(
    responses_sorted,
    stimulus_labels,
    sample_classes,
    n_exple_per_class,
    n_new,
    seed=0,
):
    """_summary_
    Takes 6 min for 20 classes with 50 sample per class

    Returns:
        tuple: _description_
    """
    # set seed for reproducibility
    random.seed(seed)

    # find shuffled labels indices to calculate
    # lower bound
    shuffled_labels = np.array(
        random.sample(
            stimulus_labels.tolist(),
            len(stimulus_labels),
        )
    )

    # find classes indices
    shuffle_classes_ix = np.array(
        [
            np.where(shuffled_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(shuffled_labels == ix)
        ]
    ).flatten()

    # get lower bound from unstructed manifolds
    # (to normalize capacity)
    shuffled = analyse_class_manifolds(
        responses_sorted[:, shuffle_classes_ix],
        shuffled_labels[shuffle_classes_ix],
        n_exple_per_class,
        n_new,
    )
    return {
        "shuffled": shuffled,
    }

    
def get_infogeometry_for_HS_and_KS(
    responses_all_sorted,
    stimulus_labels,
    sample_classes,
    n_exple_per_class,
    n_new,
    seed=0,
):
    """_summary_
    Takes 6 min for 20 classes with 50 sample per class

    Returns:
        tuple: _description_
    """
    # set seed for reproducibility
    random.seed(seed)

    # find actual labels indices
    classes_ix = np.array(
        [
            np.where(stimulus_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(stimulus_labels == ix)
        ]
    ).flatten()

    # find shuffled labels indices to calculate
    # lower bound
    shuffled_labels = np.array(
        random.sample(
            stimulus_labels.tolist(),
            len(stimulus_labels),
        )
    )

    # find classes indices
    shuffle_classes_ix = np.array(
        [
            np.where(shuffled_labels == ix)[0].tolist()
            for ix in sample_classes
            if any(shuffled_labels == ix)
        ]
    ).flatten()

    # analyse manifold information geometry
    out_allsorted = analyse_class_manifolds(
        responses_all_sorted[:, classes_ix],
        stimulus_labels[classes_ix],
        n_exple_per_class,
        n_new,
    )

    # get lower bound from unstructed manifolds (to normalize capacity)
    out_allsorted_unstructured = analyse_class_manifolds(
        responses_all_sorted[:, shuffle_classes_ix],
        shuffled_labels[shuffle_classes_ix],
        n_exple_per_class,
        n_new,
    )
    return {
        "all_sorted_units": [out_allsorted, out_allsorted_unstructured],
    }


def get_min_sample_size(sorter, class_data, min_nb_units):

    n_units = []
    for _, sorter_i in enumerate(sorter):
        for _, u_class in enumerate(class_data[sorter_i]):
            n_units.append(len(class_data[sorter_i][u_class]["unit_id"]))
    return min(np.array(n_units)[np.array(n_units) >= min_nb_units])


def get_igeom_metrics_bootstrapped(sorter: str,
                                   sorting_path: str,
                                   quality_path: str,
                                   stimulus_intervals_ms,
                                   params: dict,
                                   nb_units: int=2,
                                   sample_size: int=None,
                                   seed: int=0,
                                   block=0,
                                   n_boot=5,
                                   temp_path="."):
    """create a distribution of average manifold capacity 
    of the direction manifolds by bootstrapping
    each unit population n_boot times.

    Args:
        sorter (str): _description_
        sorting_path (str): _description_
        quality_path (str): _description_
        stimulus_intervals_ms (_type_): _description_
        params (dict): _description_
        nb_units (int, optional): minimum number of units in condition
        - for inclusion, 2 includes all with at least 2 units. capacity 
        calculated from 1 unit.
        meaningless
        - Defaults to 0.
        sample_size (int): _description_
        seed (int, optional): _description_. Defaults to 0.
        block (int): id of the run on the cluster. Each run 
        - is launched on 7 nodes on the cluster (one per 
        - spike sorter) and contains n_boot bootstrapped.
        - the dataframe output of different runs can be 
        concatenated to increase the bootstrapped sample size.

    Returns:
        _type_: _description_
    """
    # enforce reproducibility
    np.random.seed(seed)
    
    # create seeds for bootstrapping for
    # this block
    seeds = np.arange(0, n_boot, 1) + block * n_boot
    
    # initialize
    out_data = dict()
    c_data = defaultdict(dict)
    
    # load pre-computed datasets
    Sorting = si.load_extractor(sorting_path)
    quality_df = pd.read_csv(quality_path)
    
    # single-unit and multi-unit exclusively
    if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
        
        # single-unit exclusively
        out_data[sorter] = get_single_unit_responses(Sorting, stimulus_intervals_ms)
        c_data[sorter]["single-unit"] = dict()
        c_data[sorter]["single-unit"]["unit_id"] = out_data[sorter]["unit_ids"]
        c_data[sorter]["single-unit"]["responses"] = out_data[sorter][
            "responses"
        ]

        # multi-unit exclusively
        out = get_multiunit_sorted_responses(Sorting, stimulus_intervals_ms)
        c_data[sorter]["multi-unit"] = dict()
        c_data[sorter]["multi-unit"]["unit_id"] = out["unit_ids"]
        c_data[sorter]["multi-unit"]["responses"] = out["responses"]

    # all sorted units
    out = get_all_sorted_responses(Sorting, stimulus_intervals_ms)
    c_data[sorter]["all sorted units"] = dict()
    c_data[sorter]["all sorted units"]["unit_id"] = out["unit_ids"]
    c_data[sorter]["all sorted units"]["responses"] = out["responses"]

    # biased classes
    u_classes = quality_df[quality_df.sorter == sorter].quality.unique()
    
    # loop over bias classes
    for _, u_class in enumerate(u_classes):

        # record the units in this class
        c_data[sorter][u_class] = dict()
        c_data[sorter][u_class]["unit_id"] = quality_df.sorted[
            (quality_df.sorter == sorter) & (quality_df.quality == u_class)
        ].values

        # record their responses for sorters with single-unit curation
        if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
            unit_loc = np.isin(
                c_data[sorter]["single-unit"]["unit_id"], c_data[sorter][u_class]["unit_id"]
            )
            c_data[sorter][u_class]["responses"] = c_data[sorter]["single-unit"]["responses"][
                unit_loc
            ]
        if sorter in ["KS", "HS"]:
            # record their responses for sorters without single-unit curation
            unit_loc = np.isin(
                c_data[sorter]["all sorted units"]["unit_id"], c_data[sorter][u_class]["unit_id"]
            )
            c_data[sorter][u_class]["responses"] = c_data[sorter]["all sorted units"]["responses"][
                unit_loc
            ]
    
    # Info geometrics ----------------------------------
    
    # calculate information geometry metrics
    igeom_data = defaultdict(dict)

    # loop over sorted unit classes
    for _, u_class in enumerate(c_data[sorter].keys()):
        
        # initialize bootstrap key
        igeom_data[sorter][u_class] = defaultdict(dict)
        
        for boot in range(n_boot):
            
            # if has the minimum unit sample size
            if len(c_data[sorter][u_class]["unit_id"]) >= nb_units:
                
                # set reproducibility
                # use random.seed not np.random.seed
                # with random.sample                    
                random.seed(seeds[boot])
                
                # fix sample size to use
                n_units = len(c_data[sorter][u_class]["unit_id"])
                if not sample_size:
                    sample_size = n_units
                
                # we want all the units.Ssampling w/o replacement 
                # from the entire population would
                # produce the same population at every bootstrap. We sample
                # with replacement from the entire population to get an estimate
                # of the variability of the average capacity.
                ix = random.choices(np.arange(0, n_units, 1).tolist(), k=sample_size)
                sampled_resp = c_data[sorter][u_class]["responses"][ix, :]

                # calculate information geometrics
                try:
                    igeom_data[sorter][u_class][boot] = get_infogeometry_for_sorter(
                        sampled_resp, **params
                    )
                except: 
                    # set to None
                    igeom_data[sorter][u_class][boot]["data"] = None
                    igeom_data[sorter][u_class][boot]["shuffled"] = None
                igeom_data[sorter][u_class][boot]["seed"] = seeds[boot]
            else:
                # set to None
                igeom_data[sorter][u_class][boot]["data"] = None
                igeom_data[sorter][u_class][boot]["shuffled"] = None
                igeom_data[sorter][u_class][boot]["seed"] = seeds[boot]

    # # save temporary dataset
    # with open(os.path.join(os.path.dirname(temp_path), f"igeom_{block}.pkl"), "wb") as handle:
    #     pickle.dump(igeom_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Record in dataframe ----------------------------------
    
    df = pd.DataFrame()
    
    # loop over sorted unit classes
    for _, u_class in enumerate(igeom_data[sorter].keys()):
        
        # bootstrap data capacity
        # to get a distribution of capacity
        # normalized by the capacity of shuffled
        # and calculate the mean and confidence
        # interval
        for boot in range(n_boot):
            
            # if data exists
            if not igeom_data[sorter][u_class][boot]["data"] is None:
            
                print("------------------ boot number:", boot, "-----------")
                
                # get data and shuffled capacity
                data_cap = igeom_data[sorter][u_class][boot]["data"]["average_metrics"][
                    "capacity"
                ]
                
                # get shuffled average capacity
                shuffled_cap = igeom_data[sorter][u_class][boot]["shuffled"][
                    "average_metrics"
                ]["capacity"]
                
                # build dataframe
                df_i = pd.DataFrame(data=[data_cap/shuffled_cap], columns=["Capacity"])
                df_i["Unit class"] = u_class
                df_i["Sorter"] = sorter
                df = pd.concat([df, df_i])
    print("Done bootstrapping.")
    return df


def get_data_by_sorter_for_igeom(K4: str,
                                K3: str,
                                K25: str,
                                K2: str,
                                KS: str,
                                HS: str,
                                quality_path: str, 
                                stimulus_intervals_ms, 
                                seed: int=0):
    """format unit responses per quality, pooling 
    across spike sorters to calculate information
    geometrics by unit quality

    Args:
        quality_path (str): _description_
        stimulus_intervals_ms (_type_): _description_
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        defaultdict(dict): _description_
    """
    # enforce reproducibility
    np.random.seed(seed)
    
    sorters = ["KS4", "KS3", "KS2.5", "KS2", "KS", "HS"]
    
    # load sortings datasets
    SortingK4 = si.load_extractor(K4)
    SortingK3 = si.load_extractor(K3)
    SortingK25 = si.load_extractor(K25)
    SortingK2 = si.load_extractor(K2)
    SortingKS = si.load_extractor(KS)
    SortingHS = si.load_extractor(HS)
    Sortings = [SortingK4, SortingK3, SortingK25, SortingK2, SortingKS, SortingHS]
    quality_df = pd.read_csv(quality_path)
    
    # all sorted units
    c_data = defaultdict(dict)
    c_data["all sorted units"] = dict()
    c_data["all sorted units"]["unit_id"] = []
    c_data["all sorted units"]["responses"] = []

    # single-unit class exclusively
    c_data["single-unit"] = dict()
    c_data["single-unit"]["unit_id"] = []
    c_data["single-unit"]["responses"] = []
    
    # multi-units class exclusively
    c_data["multi-unit"] = dict()
    c_data["multi-unit"]["unit_id"] = []
    c_data["multi-unit"]["responses"] = []
    
    # record responses per unit class across
    # spike sorters
    for s_i, sorter in enumerate(sorters):
    
        # record all units
        d_i = get_all_sorted_responses(Sortings[s_i], stimulus_intervals_ms)
        c_data["all sorted units"]["unit_id"].append(d_i["unit_ids"])
        c_data["all sorted units"]["responses"].append(d_i["responses"])

        # record single and multi when available
        if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
        
            # single unit
            d_i = get_single_unit_responses(Sortings[s_i], stimulus_intervals_ms)
            c_data["single-unit"]["unit_id"].append(d_i["unit_ids"])
            c_data["single-unit"]["responses"].append(d_i["responses"])

            # multi-unit
            d_i = get_multiunit_sorted_responses(Sortings[s_i], stimulus_intervals_ms)
            c_data["multi-unit"]["unit_id"].append(d_i["unit_ids"])
            c_data["multi-unit"]["responses"].append(d_i["responses"])
    
    # get quality and bias classes
    u_classes = quality_df.quality.unique()
            
    # loop over bias classes
    for _, u_class in enumerate(u_classes):
        
        c_data[u_class] = dict()
        c_data[u_class]["unit_id"] = []
        c_data[u_class]["responses"] = []

        # loop over sorters
        for s_i, sorter in enumerate(sorters):
            
            # get qualified sorted units
            units = quality_df.sorted[
                (quality_df.sorter == sorter) & (quality_df.quality == u_class)
            ].values
            
            # if data exists
            if not len(units) == 0:
            
                # record data
                responses = get_unit_responses_for(units, Sortings[s_i], stimulus_intervals_ms)
                c_data[u_class]["unit_id"].append(units)
                c_data[u_class]["responses"].append(responses)

        # concatenate over sorters
        c_data[u_class]["unit_id"] = np.hstack(c_data[u_class]["unit_id"])
        c_data[u_class]["responses"] = np.vstack(c_data[u_class]["responses"])

    # concatenate over sorters
    c_data["all sorted units"]["unit_id"] = np.hstack(c_data["all sorted units"]["unit_id"])
    c_data["all sorted units"]["responses"] = np.vstack(c_data["all sorted units"]["responses"])
    c_data["single-unit"]["unit_id"] = np.hstack(c_data["single-unit"]["unit_id"])
    c_data["single-unit"]["responses"] = np.vstack(c_data["single-unit"]["responses"])
    c_data["multi-unit"]["unit_id"] = np.hstack(c_data["multi-unit"]["unit_id"])
    c_data["multi-unit"]["responses"] = np.vstack(c_data["multi-unit"]["responses"])
    return c_data


def get_igeom_metrics_by_quality_bootstrapped( 
                                   K4: str,
                                   K3: str,
                                   K25: str,
                                   K2: str,
                                   KS: str,
                                   HS: str,
                                   quality_path: str,
                                   stimulus_intervals_ms,
                                   params:dict,
                                   nb_units:int,
                                   sample_size:int,
                                   seed: int=0,
                                   block=0,
                                   n_boot=5,
                                   temp_path="."):
    
    seeds = np.arange(0, n_boot, 1)
    
    c_data = get_data_by_sorter_for_igeom(K4,
                                          K3,
                                          K25,
                                          K2,
                                          KS,
                                          HS,
                                          quality_path,
                                          stimulus_intervals_ms,
                                          seed)
    
    # calculate information geometry metrics
    igeom_data = defaultdict(dict)

    # loop over sorted unit classes
    for _, u_class in enumerate(c_data.keys()):
        
        # initialize bootstrap key
        igeom_data[u_class] = defaultdict(dict)
        
        for boot in range(n_boot):
            
            # if has the minimum unit sample size
            if len(c_data[u_class]["unit_id"]) >= nb_units:
                
                # set reproducibility
                # use random.seed not np.random.seed
                # with random.sample    
                random.seed(seeds[boot])
                
                # fix sample size to use
                n_units = len(c_data[u_class]["unit_id"])
                ix = random.choices(range(0, n_units), k=sample_size)
                sampled_resp = c_data[u_class]["responses"][ix, :]

                # calculate information geometrics
                igeom_data[u_class][boot] = get_infogeometry_for_sorter(
                    sampled_resp, **params
                )
                igeom_data[u_class][boot]["seed"] = seeds[boot]
            else:
                # set to None
                igeom_data[u_class][boot]["data"] = None
                igeom_data[u_class][boot]["shuffled"] = None
                igeom_data[u_class][boot]["seed"] = seeds[boot]

    # Record in dataframe ----------------------------------

    df = pd.DataFrame()
    
    # loop over sorted unit classes
    for _, u_class in enumerate(igeom_data.keys()):
        
        for boot in range(n_boot):
            
            # if data exists
            if not igeom_data[u_class][boot]["data"] is None:
            
                print("------------------ boot number:", boot, "-----------")
                
                # get data and shuffled capacity
                data_cap = igeom_data[u_class][boot]["data"]["average_metrics"][
                    "capacity"
                ]
                
                # get shuffled average capacity
                shuffled_cap = igeom_data[u_class][boot]["shuffled"][
                    "average_metrics"
                ]["capacity"]
                
                # build dataframe
                df_i = pd.DataFrame(data=[data_cap/shuffled_cap], columns=["Capacity"])
                df_i["Unit class"] = u_class
                df = pd.concat([df, df_i])
    return df


def _get_scores(
    SortingTrue,
    Sorting,
    delta_time: float,
):
    """get agreement scores between 
    ground truth and sorted units

    Args:
        SortingTrue (_type_): _description_
        Sorting (_type_): _description_
        delta_time (float): _description_

    Returns:
        pd.DataFrame: agreemen scores
    """
    comp = comparison.compare_sorter_to_ground_truth(
        SortingTrue,
        Sorting,
        exhaustive_gt=True,
        delta_time=delta_time,
    )
    return comp.agreement_scores


def get_sorted_unit_synapse_class(sorting_path: str, gt_path: str, delta_time=1.3):

    # load SortingExtractors
    Sorting = si.load_extractor(sorting_path)
    SortingTrue = si.load_extractor(gt_path)

    # get agreement scores
    scores = _get_scores(
        SortingTrue,
        Sorting,
        delta_time=delta_time,
    )

    # unit-test
    all(SortingTrue.unit_ids == scores.index), "assert unit ids must match"

    # locate ground truth that maximizes the score of each sorted unit
    gt_argmax_loc = np.argmax(scores.values, axis=0)
    return SortingTrue.get_property("synapse_class")[gt_argmax_loc]


def locate_unit_for_in_givenby(this_type: pd.DataFrame,
                             df_unit: pd.DataFrame,
                             type_count, 
                             seed=None,
                             with_replacement=False):
    """locate the indices of the N first units
    that match a unit type. N is given by type_count

    Args:
        this_type (pd.DataFrame): 1 row x N cols unit features
        df_unit (_typd.DataFramepe_): ground truth unit (rows) x features (col)
        type_count (_type_): number of occurence of this type among sorted units
        seed (list): seed to reorder indices if bootstrapping
        to sample a different set of ground truth units with 
        the same unit type distribution

    Returns:
        _type_: index location of ground truth in df_unit
    """
    # detect the unit type
    flagged_type = df_unit.T.apply(lambda row: all(row.values == this_type)).values
    
    # locate it
    type_loc = np.where(flagged_type)[0]
    n_types = len(type_loc)
    
    # random sample unit ids of this type
    # to match its count in the sorted population    
    if isinstance(seed, int):
        random.seed(seed)
        # method 1: sample without replacement
        try:
            ix = random.sample(np.arange(0, n_types, 1).tolist(), type_count)
        except:
            # if type_count > n_types, sample with replacement
            ix = random.choices(np.arange(0, n_types, 1).tolist(), k=type_count)            

        # method 2: sample with replacement
        if with_replacement:
            ix = random.choices(np.arange(0, n_types, 1).tolist(), k=type_count)            
            
        type_loc = type_loc[ix]
    
    # take first type_count units
    # method 1: sample without replacement
    gt_loc = type_loc.tolist()
    # method 2: shuffle without replacement
    # gt_loc = type_loc[:type_count].tolist()
    return gt_loc


def sample_gt_based_on_sorting_distribution(
    sorter, Sorting, SortingTrue, quality_path, dt, seed=None
):
    """sample ground truth units to match sorted single-unit
    distribution. We get N ground truth units from each type
    to match its count in the sorted unit population

    Args:
        sorter (_type_): _description_
        Sorting (_type_): _description_
        SortingTrue (_type_): _description_
        quality_path (_type_): _description_
        dt (_type_): _description_
        seed: seed to reorder indices in case of bootstrapping

    Returns:
        _type_: _description_
    """

    unique_type_feat = ["layer", "synapse", "etype"]
    df_gt = feat.get_unit_features(SortingTrue)
    
    # get unique types
    unique_type = df_gt.drop_duplicates()
    
    # count sorted unit types
    df = feat.get_feature_data_for(sorter, Sorting, SortingTrue, quality_path, dt)
    counts = feat.count_unit_type(
        df[unique_type_feat],
        unique_type,
    )

    # sample so as to match sorted single-unit distribution
    # get N ground truth units from that type given by count_k4
    gt_loc = []
    for ix in range(len(unique_type)):
        gt_loci = locate_unit_for_in_givenby(
            unique_type.iloc[ix], df_gt, counts.iloc[ix]["count"], seed=seed
        )
        gt_loc += gt_loci
    gt_id = df_gt.index[gt_loc]

    return {
        "gt_loc": gt_loc,
        "gt_id": gt_id,
        "df_gt": df_gt,
        "counts": counts,
        "unique_type": unique_type,
    }


def sample_pop_based_on_ref_distribution(
    SortingRef, SortingTarg, seed:int, with_replacement=False
):
    """sample ground truth units to match sorted single-unit
    distribution. We get N ground truth units from each type
    to match its count in the sorted unit population
    
    The unit types space must be the same to enable an exact 
    match.

    Args:
        sorter (_type_): _description_
        Sorting (_type_): _description_
        SortingTrue (_type_): _description_
        quality_path (_type_): _description_
        dt (_type_): _description_
        seed: seed to reorder indices in case of bootstrapping

    Returns:
        _type_: _description_
    """
    
    
    # load data
    unique_type_feat = ["layer", "synapse", "etype"]
    df_ref = feat.get_unit_features(SortingRef)
    df_targ = feat.get_unit_features(SortingTarg)
        
    # get unique types
    unique_type_targ = df_targ.drop_duplicates()
    unique_type_ref = df_ref.drop_duplicates()

    # get the common unit types, keep indices
    unique_type = pd.merge(unique_type_ref, unique_type_targ)
    df_ref.reset_index().merge(unique_type, how="left").set_index("index")
    df_targ = df_targ.reset_index().merge(unique_type, how="left").set_index("index")

    # count sorted unit typess
    count_targ = feat.count_unit_type(
        df_targ[unique_type_feat],
        unique_type,
    )

    # sample so as to match sorted single-unit distribution
    ref_loc = []
    for ix in range(len(unique_type)):
        ref_loci = locate_unit_for_in_givenby(
            unique_type.iloc[ix], df_ref, count_targ.iloc[ix]["count"], 
            seed=seed, with_replacement=with_replacement
        )
        ref_loc += ref_loci
        
    # record unit ids
    ref_id = df_ref.index[ref_loc]
    targ_id = df_targ.index
    
    # new reference population
    df_ref = df_ref.loc[ref_id]
    
    # new count of reference population
    count_ref = feat.count_unit_type(
        df_ref[unique_type_feat],
        unique_type,
    )

    # sample target units too so as to match the target unit
    # distribution
    targ_loc = []
    for ix in range(len(unique_type)):
        targ_loci = locate_unit_for_in_givenby(
            unique_type.iloc[ix], df_targ, count_targ.iloc[ix]["count"], 
            seed=seed, with_replacement=with_replacement
        )
        targ_loc += targ_loci

    # record unit ids
    targ_id = df_targ.index[targ_loc]
    
    # new reference population
    df_targ = df_targ.loc[targ_id]
    
    # new count of reference population
    count_targ = feat.count_unit_type(
        df_targ[unique_type_feat],
        unique_type,
    )    
    return {
        "targ_id": targ_id,
        "ref_id": ref_id,
        "df_targ": df_targ,
        "df_ref": df_ref,
        "count_targ": count_targ,
        "count_ref": count_ref,
        "unique_type": unique_type,
    }
    
    
def sample_gt_randomly_to_match_sorting_sample_size(
    sorter, SortingTrue, quality_path, seed:int=0
):
    """sample ground truth units to match sorted single-unit
    distribution. We get N ground truth units from each type
    to match its count in the sorted unit population

    Args:
        sorter (_type_): _description_
        Sorting (_type_): _description_
        SortingTrue (_type_): _description_
        quality_path (_type_): _description_
        dt (_type_): _description_
        seed: seed to reorder indices in case of bootstrapping

    Returns:
        _type_: _description_
    """
    
    # setup reproducibility
    # use random.seed not np.random.seed for
    # random.sample
    random.seed(seed)
    
    unique_type_feat = ["layer", "synapse", "etype"]
    df_gt = feat.get_unit_features(SortingTrue)
    
    # get unique types
    unique_type = df_gt.drop_duplicates()
    
    # count the number of single-units qualified for 
    # this sorter
    q_df = pd.read_csv(quality_path)
    n_single_units = q_df[q_df.sorter == sorter].shape[0]
    
    # randomly sample N gt
    # - method 1 without replacement
    ix = random.sample(np.arange(0, len(df_gt), 1).tolist(), n_single_units)
    # - method 2: shuffling, without replacement
    #ix = np.random.permutation(np.arange(0, n_single_units, 1))
    df_gt2 = df_gt.iloc[ix]

    # count
    this_count = feat.count_unit_type(
        df_gt2[unique_type_feat],
        unique_type,
    )
    return {
        "df_gt": df_gt2,
        "counts": this_count,
        "unique_type": unique_type,
    }


def get_stats_of_gt_randomly_sampled_to_match_sorting_sample_size(
    sorter, SortingTrue, quality_path, seeds=[0]
):
    """sample ground truth units to match sorted single-unit
    sample size and calculate probability of each unit type
    over bootstrapped samples.

    Args:
        sorter (_type_): _description_
        SortingTrue (_type_): _description_
        quality_path (_type_): _description_
        seeds: seeds

    Returns:
        _type_: _description_
    """
    counts = []
    # loop over seeds
    for seed_i in seeds:
        data = sample_gt_randomly_to_match_sorting_sample_size(
            sorter, SortingTrue, quality_path, seed_i
        )
        counts.append(data["counts"]["count"])
        
    # calculate probability of each unit type
    # over all bootstrapped samples
    stats = np.mean(counts, axis=0)
    stats /= sum(stats)
    return {
        "stats": stats,
        "counts": counts,
        "unique_type": data["unique_type"],
    }
    
    
def get_good_and_biased_sortingExtractors(sorter: str, Sorting, quality_path: str):
    """Create two SortingExtractors with good and biased units of a spike sorter
    from the qualified units data table in quality_ppath.

    Args:
        sorter (str): _description_
        Sorting (_type_): _description_
        quality_path (str): _description_

    Returns:
        _type_: _description_
    """

    q_df = pd.read_csv(quality_path)

    # biased units
    b_unit_ids = q_df[
        (q_df.sorter == sorter) & (q_df.quality == "mixed: overmerger + oversplitter")
    ].sorted

    # good units
    g_unit_ids = q_df[
        (q_df.sorter == sorter)
        & (q_df.quality == "mixed: good + overmerger + oversplitter")
    ].sorted

    # create sorting data for good and biased units
    SortingG = Sorting.select_units(unit_ids=g_unit_ids)
    SortingB = Sorting.select_units(unit_ids=b_unit_ids)

    # load data for biased units

    # unit-test
    assert len(np.unique(g_unit_ids)) == len(g_unit_ids), "unit ids should be unique"
    return SortingG, SortingB


def get_unit_ids_of_matched_unit_type_distributions(
    sorter: str, sorting_path: str, sorting_true_path: str, 
    quality_path: str, dt: float, seed:int, with_replacement=False
):
    """get unit ids of the matched unit type distributions

    Args:
        sorter (str): _description_
        sorting_path (str): _description_
        sorting_true_path (str): _description_
        quality_path (str): _description_
        dt (float): _description_
        seed (int): is required

    Returns:
        _type_: _description_
    """

    # (3m) load SortingExtractors
    Sorting = si.load_extractor(sorting_path)
    SortingTrue = si.load_extractor(sorting_true_path)

    # set sorted unit biophysical features features as properties
    Sorting = feat.set_sorted_unit_features(Sorting, SortingTrue, dt)

    # load Sorting data separated by good and biased sorted single units
    Sorting_G, Sorting_B = get_good_and_biased_sortingExtractors(
        sorter, Sorting, quality_path
    )

    # match the number of occurrences of each unique unit types
    # between reference (biased) and target (good) populations
    k4 = sample_pop_based_on_ref_distribution(
        SortingRef=Sorting_B, SortingTarg=Sorting_G, 
        seed=seed, with_replacement=with_replacement
    )
    return k4

# By spike sorter nodes ----------------

def get_igeom_metrics(sorter: str, sorting_path: str, quality_path: str, stimulus_intervals_ms, params:dict, nb_units:int, sample_size:int, seed:int=0):
    """calculate the information geometry metrics
    of the direction class manifolds

    Args:
        sorter (str): _description_
        sorting_path (str): _description_
        quality_path (str): _description_
        stimulus_intervals_ms (_type_): _description_
        params (dict): _description_
        nb_units (int): _description_
        sample_size (int): _description_
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # enforce reproducibility
    np.random.seed(seed)
    
    # initialize
    out_data = dict()
    c_data = defaultdict(dict)
    
    # load pre-computed datasets
    Sorting = si.load_extractor(sorting_path)
    quality_df = pd.read_csv(quality_path)
    
    # single-unit and multi-unit exclusively
    if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
        
        # single-unit exclusively
        out_data[sorter] = get_single_unit_responses(Sorting, stimulus_intervals_ms)
        c_data[sorter]["single-unit"] = dict()
        c_data[sorter]["single-unit"]["unit_id"] = out_data[sorter]["unit_ids"]
        c_data[sorter]["single-unit"]["responses"] = out_data[sorter][
            "responses"
        ]

        # multi-unit exclusively
        out = get_multiunit_sorted_responses(Sorting, stimulus_intervals_ms)
        c_data[sorter]["multi-unit"] = dict()
        c_data[sorter]["multi-unit"]["unit_id"] = out["unit_ids"]
        c_data[sorter]["multi-unit"]["responses"] = out["responses"]

    # all sorted units
    out = get_all_sorted_responses(Sorting, stimulus_intervals_ms)
    c_data[sorter]["all sorted units"] = dict()
    c_data[sorter]["all sorted units"]["unit_id"] = out["unit_ids"]
    c_data[sorter]["all sorted units"]["responses"] = out["responses"]

    # biased classes
    u_classes = quality_df[quality_df.sorter == sorter].quality.unique()
    
    # loop over bias classes
    for _, u_class in enumerate(u_classes):

        # record the units in this class
        c_data[sorter][u_class] = dict()
        c_data[sorter][u_class]["unit_id"] = quality_df.sorted[
            (quality_df.sorter == sorter) & (quality_df.quality == u_class)
        ].values

        # record their responses for sorters with single-unit curation
        if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
            unit_loc = np.isin(
                c_data[sorter]["single-unit"]["unit_id"], c_data[sorter][u_class]["unit_id"]
            )
            c_data[sorter][u_class]["responses"] = c_data[sorter]["single-unit"]["responses"][
                unit_loc
            ]
        if sorter in ["KS", "HS"]:
            # record their responses for sorters without single-unit curation
            unit_loc = np.isin(
                c_data[sorter]["all sorted units"]["unit_id"], c_data[sorter][u_class]["unit_id"]
            )
            c_data[sorter][u_class]["responses"] = c_data[sorter]["all sorted units"]["responses"][
                unit_loc
            ]
        
    # examples
    print("Spike sorter:", c_data.keys())
    print("Unit classes:", c_data[sorter].keys())
    
    # Info geometrics ----------------------------------
    
    # calculate information geometry metrics
    igeom_data = defaultdict(dict)
    
    # loop over sorted unit classes
    for _, u_class in enumerate(c_data[sorter].keys()):

        # if has the minimum unit sample size
        if len(c_data[sorter][u_class]["unit_id"]) >= nb_units:

            # fix sample size to use
            ix = np.random.permutation(np.arange(0, sample_size, 1))
            sampled_resp = c_data[sorter][u_class]["responses"][ix, :]

            # calculate information geometrical metrics
            igeom_data[sorter][u_class] = get_infogeometry_for_sorter(
                sampled_resp, **params
            )
        else:
            # set to None
            igeom_data[sorter][u_class] = defaultdict(dict)
            igeom_data[sorter][u_class]["data"] = None
            igeom_data[sorter][u_class]["shuffled"] = None

    # Record in dataframe ----------------------------------
    df = pd.DataFrame()
    
    # loop over sorted unit classes
    for _, u_class in enumerate(igeom_data[sorter].keys()):
        if not igeom_data[sorter][u_class]["data"] is None:

            # get data and shuffled capacity
            data_cap = igeom_data[sorter][u_class]["data"]["metrics_per_class"][
                "capacity"
            ]
            shuffled_cap = igeom_data[sorter][u_class]["shuffled"][
                "metrics_per_class"
            ]["capacity"]

            # build dataframe
            df_i = pd.DataFrame(data=data_cap / np.mean(shuffled_cap), columns=["Capacity"])
            df_i["Unit class"] = u_class
            df_i["Sorter"] = sorter
            df = pd.concat([df, df_i])
    return df


def get_igeom_metrics_for_sorter_bootstrapped(sorter: str,
                                   sorting_path: str,
                                   stimulus_intervals_ms,
                                   params: dict,
                                   block=0,
                                   n_boot=5):
    """Calculate the information geometrics (manifold capacity, radius, 
    direction dimensionality, centroid correlations, K)
    for sorter "sorter", by bootstrapping the shuffled responses
    to obtain a distribution of normalized capacity values.
    
    Note that the neural response capacity is the same value
    over bootstraps. Only shuffled capacity changes.
    
    Args:
        sorter (str): sorter: "KS4", "KS3", "KS2.5, "KS2", "KS", "HS" 
        - as found in dataframe stored at quality_path under the column "sorter"
        sorting_path (str): path of the SortingExtractor
        
        quality_path (str): path of the csv file containing sorted single units' 
        - quality dataset
        
        stimulus_intervals_ms (_type_): _description_
        
        params (dict): _description_
        
        nb_units (int, optional): minimum number of units in condition
        - for inclusion, 2 includes all with at least 2 units. capacity 
        calculated from 1 unit.
        meaningless
        - Defaults to 0.
        
        sample_size (int): number of units to sample
        
        block (int): id of the run on the cluster. Each run 
        - is launched on 7 nodes on the cluster (one per 
        - spike sorter) and contains n_boot bootstrapped.
        - the dataframe output of different runs can be 
        concatenated to increase the bootstrapped sample size.

    Returns:
        pd.DataFrame: capacities for each boostrapped sample
        by unit quality class (good and biased)
    """
    # create seeds for bootstrapping in this
    # this run block
    seeds = np.arange(0, n_boot, 1) + block * n_boot
    
    # preallocate dataset  
    out_data = dict()
    c_data = defaultdict(dict)
    
    # load pre-computed datasets
    Sorting = si.load_extractor(sorting_path)
    
    # get unit response data
    if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
        # get single-unit data
        out_data = get_single_unit_responses(Sorting, stimulus_intervals_ms)
    elif sorter in ["KS", "HS"]:
        # get all unit data because these sorters do not curate single-units
        out_data = get_all_sorted_responses(Sorting, stimulus_intervals_ms)
    else:
        raise ValueError("sorter must either be KS4, KS3, KS2.5, KS2, KS or HS")
    c_data["unit_id"] = out_data["unit_ids"]
    c_data["responses"] = out_data["responses"]
        
    # Calculate info geometrics ----------------------------------
    
    # preallocate datasets
    igeom_data = dict()
    df = pd.DataFrame()
    df_c = pd.DataFrame()

    # bootstrap to get different shuffled responses
    # and a distribution of normalized capacity
    # neural data capacity is identical for all bootstraps
    # [TODO]: this can be sped up by calculating capacity
    # once for the neural responses and for the
    # shuffled responses in the bootstrapping loop
    for boot in range(n_boot):
        
        # set a different seed for shuffling and 
        # dimensionality reduction per bootstrap
        params["seed_dim_red"] = seeds[boot]
        params["seed_shuffling"] = seeds[boot]

        # calculate information geometrics
        try:
            igeom_data[boot] = get_infogeometry_for_sorter(
                c_data["responses"], **params
            )
        except:
            igeom_data[boot]["data"] = None
            igeom_data[boot]["shuffled"] = None
        
        # record the seed
        igeom_data[boot]["seed"] = seeds[boot]
        igeom_data[boot]["seed"] = seeds[boot]

        # if metrics exists
        if not igeom_data[boot]["data"] is None:
            
            # get data average capacity
            data_cap = igeom_data[boot]["data"]["average_metrics"][
                "capacity"
            ]
            
            # get the shuffled response average capacity
            shuffled_cap = igeom_data[boot]["shuffled"][
                "average_metrics"
            ]["capacity"]

            # get geometrics
            data_radius = igeom_data[boot]["data"]["average_metrics"][
                "radius"
            ]
            data_dims = igeom_data[boot]["data"]["average_metrics"][
                "dimensions"
            ]
            data_corr = igeom_data[boot]["data"]["average_metrics"][
                "correlation"
            ]
            data_k = igeom_data[boot]["data"]["average_metrics"][
                "K"
            ]

        # iteratively build the dataset
        df_c["Capacity"] = [data_cap / shuffled_cap]
        
        # record geometrics
        df_c["Radius"] = [data_radius]
        df_c["Dimension"] = [data_dims]
        df_c["Correlation"] = [data_corr]
        df_c["K"] = [data_k]

        # record metadata
        df_c["Unit class"] = "sorted_unit"
        df_c["Sorter"] = sorter
        df_c["Sampling scheme"] = "None"
        df = pd.concat([df, df_c])
    print("Done bootstrapping.")
    return df


def get_igeom_metrics_for_thresh_crossing_bootstrapped(
                                   sorting_path: str,
                                   stimulus_intervals_ms,
                                   params: dict,
                                   block=0,
                                   n_boot=5):
    """Calculate the information geometrics (manifold capacity, radius, 
    direction dimensionality, centroid correlations, K)
    for sorter "sorter", by bootstrapping the shuffled responses
    to obtain a distribution of normalized capacity values.
    
    Note that the neural response capacity is the same value
    over bootstraps. Only shuffled capacity changes.
    
    Args:
        sorting_path (str): path of the SortingExtractor
        
        quality_path (str): path of the csv file containing sorted single units' 
        - quality dataset
        
        stimulus_intervals_ms (_type_): _description_
        
        params (dict): _description_
        
        nb_units (int, optional): minimum number of units in condition
        - for inclusion, 2 includes all with at least 2 units. capacity 
        calculated from 1 unit.
        meaningless
        - Defaults to 0.
        
        sample_size (int): number of units to sample
        
        block (int): id of the run on the cluster. Each run 
        - is launched on 7 nodes on the cluster (one per 
        - spike sorter) and contains n_boot bootstrapped.
        - the dataframe output of different runs can be 
        concatenated to increase the bootstrapped sample size.

    Returns:
        pd.DataFrame: capacities for each boostrapped sample
        by unit quality class (good and biased)
    """
    # create seeds for bootstrapping in this
    # this run block
    seeds = np.arange(0, n_boot, 1) + block * n_boot
    
    # preallocate dataset  
    out_data = dict()
    c_data = defaultdict(dict)
    
    # load pre-computed datasets
    Sorting = si.load_extractor(sorting_path)
    
    # curate spikes (sites) in cortex
    is_in_ctx = Sorting.get_property('layer')!='Outside'
    spikes_in_cortex = Sorting.unit_ids[is_in_ctx]
    Sorting = Sorting.select_units(unit_ids=spikes_in_cortex)

    # get all unit data because these sorters do not curate single-units
    out_data = get_all_sorted_responses(Sorting, stimulus_intervals_ms)

    c_data["unit_id"] = out_data["unit_ids"]
    c_data["responses"] = out_data["responses"]
        
    # Calculate info geometrics ----------------------------------
    
    # preallocate datasets
    igeom_data = dict()
    df = pd.DataFrame()
    df_c = pd.DataFrame()

    # bootstrap to get different shuffled responses
    # and a distribution of normalized capacity
    # neural data capacity is identical for all bootstraps
    # [TODO]: this can be sped up by calculating capacity
    # once for the neural responses and for the
    # shuffled responses in the bootstrapping loop
    for boot in range(n_boot):
        
        # set a different seed for shuffling and 
        # dimensionality reduction per bootstrap
        params["seed_dim_red"] = seeds[boot]
        params["seed_shuffling"] = seeds[boot]

        # calculate information geometrics
        try:
            igeom_data[boot] = get_infogeometry_for_sorter(
                c_data["responses"], **params
            )
        except:
            igeom_data[boot]["data"] = None
            igeom_data[boot]["shuffled"] = None
        
        # record the seed
        igeom_data[boot]["seed"] = seeds[boot]
        igeom_data[boot]["seed"] = seeds[boot]

        # if metrics exists
        if not igeom_data[boot]["data"] is None:
            
            # get data average capacity
            data_cap = igeom_data[boot]["data"]["average_metrics"][
                "capacity"
            ]
            
            # get the shuffled response average capacity
            shuffled_cap = igeom_data[boot]["shuffled"][
                "average_metrics"
            ]["capacity"]

            # get geometrics
            data_radius = igeom_data[boot]["data"]["average_metrics"][
                "radius"
            ]
            data_dims = igeom_data[boot]["data"]["average_metrics"][
                "dimensions"
            ]
            data_corr = igeom_data[boot]["data"]["average_metrics"][
                "correlation"
            ]
            data_k = igeom_data[boot]["data"]["average_metrics"][
                "K"
            ]

        # iteratively build the dataset
        df_c["Capacity"] = [data_cap / shuffled_cap]
        
        # record geometrics
        df_c["Radius"] = [data_radius]
        df_c["Dimension"] = [data_dims]
        df_c["Correlation"] = [data_corr]
        df_c["K"] = [data_k]

        # record metadata
        df_c["Unit class"] = "sorted_unit"
        df_c["Sorter"] = "thresh-crossing"
        df_c["Sampling scheme"] = "None"
        df = pd.concat([df, df_c])
    print("Done bootstrapping.")
    return df


# By sampling bias nodes ---------------

def get_igeom_metrics_bootstrapped_for_ground_truth(stimulus_intervals_ms, 
                                                    sorting_true_path, 
                                                    params: dict,
                                                    sample_size=None,
                                                    block=0, 
                                                    n_boot=5):
            
    """Calculate the information geometrics (manifold capacity, radius, 
    direction dimensionality, centroid correlations, K) of the direction
    manifolds for the ground truth.
    
    The information capacity is normalized by the lower bound.
    The lower bound is the information capacity of the 
    direction-shuffled responses, averaged over all bootstraps.
    Calculating one lower bound averaged over bootstrapped is much faster to 
    compute and reduces variability (e.g., if data info capacity is fixed if
    we take the entire dataset, all variability would be produced by
    the different lower bounds produced by shuffled samples of the dataset).
    By choosing a unique, averaged lower bound we discard that variability.
    
    Args:
        sorter (str): _description_
        sorting_path (str): _description_
        quality_path (str): _description_
        stimulus_intervals_ms (_type_): _description_
        params (dict): _description_
        nb_units (int or None): _description_
        sample_size (int): 
        - None: get all units
        - or int: subsample without replacement
        - or "all": sample with replacement from all units 
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
                
    np.random.seed(0)
    
    # create seeds for bootstrapping for this block
    seeds = np.arange(0, n_boot, 1) + block * n_boot
    
    # load ground truth sortingExtractor
    SortingTrue = si.load_extractor(sorting_true_path)
            
    # compute unit x stimulus response matrix
    gt_resp = compute_response_by_stim_matrix(
        SortingTrue.unit_ids, SortingTrue, stimulus_intervals_ms
    )
    
    # build dataset by unit class
    c_data = defaultdict(dict)
    c_data["ground_truth"]["ground_truth"] = dict()
    c_data["ground_truth"]["ground_truth"]["unit_id"] = SortingTrue.unit_ids
    c_data["ground_truth"]["ground_truth"]["responses"] = gt_resp

    data_cap = []
    shuffled_cap = []
    data_radius = []
    data_dims = []
    data_corr = []
    data_k = []
    
    # loop over bootstrapped samples
    for boot in range(n_boot):
        
        # setup reproducibility
        # use random.seed not np.random.seed
        # with random.sample
        random.seed(seeds[boot])

        # !testing!
        params["seed_dim_red"] = seeds[boot]

        # sample units
        n_units = len(c_data["ground_truth"]["ground_truth"]["unit_id"])
        
        if sample_size:
            if sample_size == "all":
                # sample with replacement. Sampling without replacement 
                # from the entire pop. would produce the same units every time
                # see else.
                ix = random.choices(range(0, n_units), k=n_units)
            else:
                # sample a subset without replacement
                ix = random.sample(np.arange(0, n_units, 1).tolist(), sample_size)
            sampled_resp = c_data["ground_truth"]["ground_truth"]["responses"][ix, :]
        else:
            # we do not sample but just get the entire population
            sampled_resp = c_data["ground_truth"]["ground_truth"]["responses"]
               
        # calculate the information geometrical metrics
        igeom_data = defaultdict(dict)
        igeom_data["ground_truth"]["ground_truth"] = dict()
        igeom_data["ground_truth"]["ground_truth"] = get_true_infogeometry(
            sampled_resp, **params
        )
        
        # get the information capacities for the data and shuffled data
        data_cap.append(igeom_data["ground_truth"]["ground_truth"]["data"]["average_metrics"][
            "capacity"
        ])
        shuffled_cap.append(igeom_data["ground_truth"]["ground_truth"]["shuffled"][
            "average_metrics"
        ]["capacity"])
        
        # get geometrics
        data_radius.append(igeom_data["ground_truth"]["ground_truth"]["data"]["average_metrics"][
            "radius"
        ])       
        data_dims.append(igeom_data["ground_truth"]["ground_truth"]["data"]["average_metrics"][
            "dimensions"
        ])        
        data_corr.append(igeom_data["ground_truth"]["ground_truth"]["data"]["average_metrics"][
            "correlation"
        ])
        data_k.append(igeom_data["ground_truth"]["ground_truth"]["data"]["average_metrics"][
            "K"
        ])

    # build the dataset of capacities
    # normalize capacity by dividing it by the average lower bound information capacity (averaged
    # over the bootstrapped samples).
    
    # record normalized capacity
    df_gt = pd.DataFrame(data=np.array(data_cap) / np.mean(shuffled_cap), columns=["Capacity"])

    # record geometrics
    df_gt["Radius"] = np.array(data_radius)
    df_gt["Dimension"] = np.array(data_dims)
    df_gt["Correlation"] = np.array(data_corr)
    df_gt["K"] = np.array(data_k)

    # record metadata
    df_gt["Unit class"] = "ground_truth"
    df_gt["Sorter"] = "ground_truth"
    df_gt["Sampling scheme"] = "random"
    print("Done bootstrapping.")
    return df_gt


def get_igeom_stats_for_gt_and_random_sampling(stimulus_intervals_ms,
                                               sorter: str,
                                               sorting_path,
                                               sorting_true_path, 
                                               quality_path,
                                               params: dict, 
                                               block=0, 
                                               n_boot=5):
            
    """calculate the information geometry metrics
    of the direction class manifolds for the ground truth
    sample size matches the number of sorted single units 

    Args:
        sorter (str): _description_
        sorting_path (str): _description_
        quality_path (str): _description_
        stimulus_intervals_ms (_type_): _description_
        params (dict): _description_
        nb_units (int): _description_
        sample_size (int): _description_
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # create seeds for bootstrapping for this block
    seeds = np.arange(0, n_boot, 1) + block * n_boot
    
    # load ground truth sortingExtractor
    SortingTrue = si.load_extractor(sorting_true_path)
    n_gt = len(SortingTrue.unit_ids)
               
    # count the number of single-units qualified for 
    # this sorter
    q_df = pd.read_csv(quality_path)
    n_single_units = q_df[q_df.sorter == sorter].shape[0]
    
    # compute unit x stimulus response matrix
    gt_resp = compute_response_by_stim_matrix(
        SortingTrue.unit_ids, SortingTrue, stimulus_intervals_ms
    )
    
    # build dataset by unit class
    c_data = defaultdict(dict)
    c_data["ground_truth"]["ground_truth"] = dict()
    c_data["ground_truth"]["ground_truth"]["unit_id"] = SortingTrue.unit_ids
    c_data["ground_truth"]["ground_truth"]["responses"] = gt_resp

    df_gt = pd.DataFrame()
    data_cap = []
    shuffled_cap = []
    
    for boot in range(n_boot):
        
        # setup reproducibility
        # use random.seed not np.random.seed
        # with random.sample
        random.seed(seeds[boot])
        
        params["seed_dim_red"] = seeds[boot]
                
        # sample ground truth units to match
        # the distribution of sorted unit types
        # the mumber of ground truth units match the 
        # number of sorted single units
        # method 1: sample without replacement -> good uniform sampling
        ix = random.sample(np.arange(0, n_gt, 1).tolist(), n_single_units)
        # method 2: shuffling without replacement
        #ix = np.random.permutation(np.arange(0, n_single_units, 1))
        sampled_resp = c_data["ground_truth"]["ground_truth"]["responses"][ix, :]
        
        # calculate information geometrical metrics
        igeom_data = defaultdict(dict)
        igeom_data["ground_truth"]["ground_truth"] = dict()
        igeom_data["ground_truth"]["ground_truth"] = get_true_infogeometry(
            sampled_resp, **params
        )
        
        # get data and shuffled data information capacity
        data_cap.append(igeom_data["ground_truth"]["ground_truth"]["data"]["average_metrics"][
            "capacity"
        ])
        shuffled_cap.append(igeom_data["ground_truth"]["ground_truth"]["shuffled"][
            "average_metrics"
        ]["capacity"])

    # build dataframe
    # normalize infocapacity by shuffled info capacity
    df_gt = pd.DataFrame(data=np.array(data_cap) / np.mean(shuffled_cap), columns=["Capacity"])
    
    # label
    df_gt["Unit class"] = "ground_truth"
    df_gt["Sampling scheme"] = "random"
    df_gt["Sorter"] = sorter
    print("Done bootstrapping.")
    return df_gt


def get_igeom_stats_for_gt_and_biased_sampling(stimulus_intervals_ms, 
                                                                        sorter,
                                                    sorting_path,                 
                                                    sorting_true_path, 
                                                    quality_path,
                                                    dt,
                                                    params: dict, 
                                                    block=0, 
                                                    n_boot=5):
            
    """calculate the information geometry metrics
    of the direction class manifolds for the ground truth

    Args:
        sorter (str): _description_
        sorting_path (str): _description_
        quality_path (str): _description_
        stimulus_intervals_ms (_type_): _description_
        params (dict): _description_
        nb_units (int): _description_
        sample_size (int): _description_
        seed (int, optional): _description_. Defaults to 0.

    Returns:
        _type_: _description_
    """
    # create seeds for bootstrapping for this block
    seeds = np.arange(0, n_boot, 1) + block * n_boot
    
    # load ground truth sortingExtractor
    SortingTrue = si.load_extractor(sorting_true_path)
    Sorting = si.load_extractor(sorting_path)
            
    # compute unit x stimulus response matrix
    gt_resp = compute_response_by_stim_matrix(
        SortingTrue.unit_ids, SortingTrue, stimulus_intervals_ms
    )
    
    # build dataset by unit class
    c_data = defaultdict(dict)
    c_data["ground_truth"]["ground_truth"] = dict()
    c_data["ground_truth"]["ground_truth"]["unit_id"] = SortingTrue.unit_ids
    c_data["ground_truth"]["ground_truth"]["responses"] = gt_resp

    df_gt = pd.DataFrame()
    data_cap = []
    shuffled_cap = []
    
    for boot in range(n_boot):
        
        
        
        # !test!
        params["seed_dim_red"] = seeds[boot]
                                       
        
        
        # sample ground truth units to match
        # the distribution of sorted unit types
        data = sample_gt_based_on_sorting_distribution(
            sorter, Sorting, SortingTrue, quality_path, dt, seeds[boot]
        )
        sampled_resp = c_data["ground_truth"]["ground_truth"]["responses"][data["gt_loc"], :]
        
        # calculate information geometrical metrics
        igeom_data = defaultdict(dict)
        igeom_data["ground_truth"]["ground_truth"] = dict()
        igeom_data["ground_truth"]["ground_truth"] = get_true_infogeometry(
            sampled_resp, **params
        )
        
        # get data and shuffled data information capacity
        data_cap.append(igeom_data["ground_truth"]["ground_truth"]["data"]["average_metrics"][
            "capacity"
        ])
        shuffled_cap.append(igeom_data["ground_truth"]["ground_truth"]["shuffled"][
            "average_metrics"
        ]["capacity"])

    # build dataframe
    # normalize infocapacity by shuffled info capacity
    df_gt = pd.DataFrame(data=np.array(data_cap) / np.mean(shuffled_cap), columns=["Capacity"])
    df_gt["Unit class"] = "ground_truth"
    df_gt["Sampling scheme"] = "biased"
    df_gt["Sorter"] = sorter
    print("Done bootstrapping.")
    return df_gt

# By unit quality nodes ---------------

def get_igeom_metrics_bootstrapped_by_sorter_and_q_wo_sampling_bias(sorter: str,
                                   sorting_path: str,
                                   sorting_true_path: str, 
                                   dt: float,
                                   quality_path: str,
                                   stimulus_intervals_ms,
                                   params: dict,
                                   nb_units: int=2,
                                   sample_size: int=None,
                                   seed: int=0,
                                   block=0,
                                   n_boot=5,
                                   temp_path="."):
    """get the distribution of information geometry information capacities 
    for spike sorter "sorter" and unit qualities (biased and good units)
    after constraining the unit type distribution to be the same between
    the unit quality types in each bootstrapped sample.
    
    Args:
        sorter (str): sorter: "KS4", "KS3", "KS2.5, "KS2", "KS", "HS" 
        - as found in dataframe stored at quality_path under the column "sorter"
        sorting_path (str): path of the SortingExtractor
        
        quality_path (str): path of the csv file containing sorted single units' 
        - quality dataset
        
        stimulus_intervals_ms (_type_): _description_
        
        params (dict): _description_
        
        nb_units (int, optional): minimum number of units in condition
        - for inclusion, 2 includes all with at least 2 units. capacity 
        calculated from 1 unit.
        meaningless
        - Defaults to 0.
        
        sample_size (int): number of units to sample
        
        seed (int, optional): _description_. Defaults to 0.
        - for bootstrapping different unit samples
        
        block (int): id of the run on the cluster. Each run 
        - is launched on 7 nodes on the cluster (one per 
        - spike sorter) and contains n_boot bootstrapped.
        - the dataframe output of different runs can be 
        concatenated to increase the bootstrapped sample size.

    Returns:
        pd.DataFrame: capacities for each boostrapped sample
        by unit quality class (good and biased)
    """
    # create seeds for bootstrapping for
    # this block
    seeds = np.arange(0, n_boot, 1) + block * n_boot
    
    # initialize
    out_data = dict()
    c_data = defaultdict(dict)
    
    # load pre-computed datasets
    Sorting = si.load_extractor(sorting_path)
    quality_df = pd.read_csv(quality_path)
    
    # single-unit and multi-unit exclusively
    if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
        
        # single-unit exclusively
        out_data[sorter] = get_single_unit_responses(Sorting, stimulus_intervals_ms)
        c_data[sorter]["single-unit"] = dict()
        c_data[sorter]["single-unit"]["unit_id"] = out_data[sorter]["unit_ids"]
        c_data[sorter]["single-unit"]["responses"] = out_data[sorter][
            "responses"
        ]

        # multi-unit exclusively
        out = get_multiunit_sorted_responses(Sorting, stimulus_intervals_ms)
        c_data[sorter]["multi-unit"] = dict()
        c_data[sorter]["multi-unit"]["unit_id"] = out["unit_ids"]
        c_data[sorter]["multi-unit"]["responses"] = out["responses"]

    # all sorted units
    out = get_all_sorted_responses(Sorting, stimulus_intervals_ms)
    c_data[sorter]["all sorted units"] = dict()
    c_data[sorter]["all sorted units"]["unit_id"] = out["unit_ids"]
    c_data[sorter]["all sorted units"]["responses"] = out["responses"]

    # biased classes
    u_classes = quality_df[quality_df.sorter == sorter].quality.unique()
    
    # loop over bias classes
    for _, u_class in enumerate(u_classes):

        # record the units in this class
        c_data[sorter][u_class] = dict()
        c_data[sorter][u_class]["unit_id"] = quality_df.sorted[
            (quality_df.sorter == sorter) & (quality_df.quality == u_class)
        ].values

        # record their responses for sorters with single-unit curation
        if sorter in ["KS4", "KS3", "KS2.5", "KS2"]:
            unit_loc = np.isin(
                c_data[sorter]["single-unit"]["unit_id"], c_data[sorter][u_class]["unit_id"]
            )
            c_data[sorter][u_class]["responses"] = c_data[sorter]["single-unit"]["responses"][
                unit_loc
            ]
        if sorter in ["KS", "HS"]:
            # record their responses for sorters without single-unit curation
            unit_loc = np.isin(
                c_data[sorter]["all sorted units"]["unit_id"], c_data[sorter][u_class]["unit_id"]
            )
            c_data[sorter][u_class]["responses"] = c_data[sorter]["all sorted units"]["responses"][
                unit_loc
            ]
    
    # Info geometrics ----------------------------------
    
    # preallocate metrics
    igeom_data = defaultdict(dict)

    # select only two unit qualities classes to contrast
    c_slct = defaultdict(dict)
    g_class = 'mixed: good + overmerger + oversplitter'
    b_class = 'mixed: overmerger + oversplitter'
    
    c_slct[sorter][g_class] = c_data[sorter][g_class]
    c_slct[sorter][b_class] = c_data[sorter][b_class]
    
    # loop over sorted unit classes
    # initialize bootstrap key
    igeom_data[sorter]['mixed: good + overmerger + oversplitter'] = defaultdict(dict)
    igeom_data[sorter]['mixed: overmerger + oversplitter'] = defaultdict(dict)
    
    # bootstrap while enforcing that good and biased unit
    # have identical unit type distributions
    for boot in range(n_boot):

        # count good and biased units
        n_g = len(c_slct[sorter][g_class]["unit_id"])
        n_b = len(c_slct[sorter][b_class]["unit_id"])
        
        # if enough of them
        if (n_g >= nb_units) and (n_b >= nb_units):
            
            # sample good and biased units with identical unit type distrution --------------
            # note: one seed per bootstrap
            sampled_units = get_unit_ids_of_matched_unit_type_distributions(
                sorter,
                sorting_path=sorting_path,
                sorting_true_path=sorting_true_path,
                quality_path=quality_path,
                dt=dt,
                seed=seeds[boot],
                with_replacement=True,
            )
            
            # good units
            g_ix = np.where(np.isin(
                c_slct[sorter][g_class]["unit_id"], sampled_units["targ_id"]
            ))[0]
            g_resp = c_slct[sorter][g_class]["responses"][g_ix, :]

            # biased units
            b_ix = np.where(np.isin(
                c_slct[sorter][b_class]["unit_id"],  sampled_units["ref_id"]
            ))[0]
            b_resp = c_slct[sorter][b_class]["responses"][b_ix, :]

            # calculate information geometrics --------------
            
            # set a different seed for shuffling and 
            # dimensionality reduction per bootstrap
            params["seed_dim_red"] = seeds[boot]
            params["seed_shuffling"] = seeds[boot]            
            
            # good units
            try:
                igeom_data[sorter][g_class][boot] = get_infogeometry_for_sorter(
                    g_resp, **params
                )
            except:
                igeom_data[sorter][g_class][boot]["data"] = None
                igeom_data[sorter][g_class][boot]["shuffled"] = None
            
            # biased units
            try:
                igeom_data[sorter][b_class][boot] = get_infogeometry_for_sorter(
                        b_resp, **params                    
                    )
            except:
                igeom_data[sorter][b_class][boot]["data"] = None
                igeom_data[sorter][b_class][boot]["shuffled"] = None
            
            # record the seed
            igeom_data[sorter][g_class][boot]["seed"] = seeds[boot]
            igeom_data[sorter][b_class][boot]["seed"] = seeds[boot]
        else:
            igeom_data[sorter][g_class][boot]["data"] = None
            igeom_data[sorter][g_class][boot]["shuffled"] = None
            igeom_data[sorter][g_class][boot]["seed"] = seeds[boot]
            igeom_data[sorter][b_class][boot]["data"] = None
            igeom_data[sorter][b_class][boot]["shuffled"] = None
            igeom_data[sorter][b_class][boot]["seed"] = seeds[boot]

    # Record in dataframe ----------------------------------
    
    df = pd.DataFrame()
    data_cap = []
    shuffled_cap = []
    u_classes = []
    
    # loop over sorted unit classes
    for _, u_class in enumerate(igeom_data[sorter].keys()):
        
        # bootstrap data capacity
        # to get a distribution of capacity
        # normalized by the capacity of shuffled
        # and calculate the mean and confidence
        # interval
        for boot in range(n_boot):
            
            # if data exists
            if not igeom_data[sorter][u_class][boot]["data"] is None:
            
                print("------------------ boot number:", boot, "-----------")
                
                # get data and shuffled capacity
                data_cap.append(igeom_data[sorter][u_class][boot]["data"]["average_metrics"][
                    "capacity"
                ])
                
                # get shuffled average capacity
                shuffled_cap.append(igeom_data[sorter][u_class][boot]["shuffled"][
                    "average_metrics"
                ]["capacity"])
                
        # build dataframe
        df_c = pd.DataFrame(data=np.array(data_cap) / np.mean(shuffled_cap), columns=["Capacity"])
        df_c["Unit class"] = u_class
        df_c["Sorter"] = sorter
        df = pd.concat([df, df_c])
    print("Done bootstrapping.")
    return df

# Tests nodes ---------------

def test_sample_pop_based_on_ref_distribution(k4_ids, k4_df, Sorting):

    assert any(np.isin(k4_ids, Sorting.unit_ids)), "there should be common unit ids"

    tests = []
    for ix in range(len(k4_ids)):
        test_df = pd.DataFrame()
        test_df["layer"] = Sorting.get_property("layer")[
            np.where(Sorting.unit_ids == k4_ids[ix])[0]
        ]
        test_df["synapse"] = Sorting.get_property("synapse_class")[
            np.where(Sorting.unit_ids == k4_ids[ix])[0]
        ]
        test_df["etype"] = Sorting.get_property("etype")[
            np.where(Sorting.unit_ids == k4_ids[ix])[0]
        ]
        tests.append(all(k4_df.loc[k4_ids[ix]].drop_duplicates().values == test_df))

    assert all(tests), "features should match for same unit ids in both datasets"