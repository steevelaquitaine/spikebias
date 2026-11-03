"""Features of the ground truth neurons and of their waveforms in synthetic (Buccino, MEArec) and biophysical simulations

Functions copied from notebooks/1_results/supp_fig6_CD_pros_of_added_details_yield_revised.ipynb (Supplementary Fig. 6c).
"""

import numpy as np
from numpy.linalg import norm as lalgnorm
from scipy.spatial.distance import cdist
import spikeinterface.qualitymetrics as sqm
import spikeinterface.core.template_tools as ttools


def _euclidean_distance(coord_1, coord_2):
    return np.sqrt(np.sum((coord_1 - coord_2) ** 2))


def mad(data):
    mean_data = np.mean(data)
    return np.mean(np.absolute(data - mean_data))


def get_mad_ratio(spike_amp, noise_amp):
    """calculate an sd_ratio robust to outliers

    Args:
        spike_amp (_type_): _description_
        noise_amp (_type_): _description_

    Returns:
        _type_: _description_
    """
    mad_unit = mad(spike_amp)  # twice smaller than std
    mad_noise = mad(noise_amp)
    return mad_unit / mad_noise


def get_best_site_mad_noise(analyzer, max_chids, unit):

    # get waveforms
    # e.g., (500 spikes, 160 samples, 18 channels)
    wv = analyzer.get_extension('waveforms').get_waveforms_one_unit(unit_id=unit)

    # get channel ids (sparse)
    channel_ids = analyzer.sparsity.unit_id_to_channel_ids[unit]
    
    # get nearest channel
    max_chid = max_chids[unit]
    max_chid_ix = np.where(channel_ids == max_chid)[0][0]
    return wv[:, :, max_chid_ix].flatten()


def get_waveform_amplitudes_per_unit(analyzer):
    return analyzer.get_extension("spike_amplitudes").get_data(outputs="by_unit")[0]


def get_cell_locations(GroundTruth):

    x = GroundTruth.get_property('x')
    y = GroundTruth.get_property('y')
    z = GroundTruth.get_property('z')
    return np.array([x, y, z]).T

# SIMULATE SYNTHETIC RECORDING WITH MEAREC


def count_near_cells_within_radius(site_pos, cell_pos, unit_ids, radius=50):
    """
    Counts the number of unique cells within a specified radius from each site.

    Args:
        recording: SpikeInterface RecordingExtractor object.
        sorting: SpikeInterface SortingExtractor object.
        radius (int): Distance threshold in micrometers.

    Returns:
        int: Number of unique cells within the specified radius.
    """
    all_near_cells = []

    # for each site, find cells within the specified radius
    for pos in site_pos:
        tiled = np.tile(pos, (cell_pos.shape[0], 1))
        dist = np.linalg.norm(cell_pos - tiled, axis=1)
        near_cells = unit_ids[dist < radius]
        all_near_cells.extend(near_cells)
    all_near_cells = list(set(all_near_cells))
    return len(all_near_cells)


# QUALITY METRICS FOR A SINGLE UNIT


def get_spatial_spread(
    analyzer, unit_id: int, max_chids: dict, channel_ids, channel_coord
):
    """measure unit's spatial spread (a.u): the spatial distance of the nearest site to the max site 
    weighted by the similarity of their waveform to the max waveform.
        
        cosine_similarity = dot(near_waveform, max_waveform) / ||near_waveform||.||max_waveform||

        rectify: if cosine_similarity < 0: cosine_similarity=0

        spread = dot(cosine_similarity, site_distance)

    note: cosine similarity ranges between 0 and 1 and the site distance is in um.

    Args:
        unit_id (int): _description_
        max_chids (dict): _description_
        Recording (_type_): _description_
        channel_ids (_type_): _description_
        channel_coord (_type_): _description_

    Returns:
        _type_: _description_
    """
    # get waveforms
    wave_extension = analyzer.get_extension('waveforms')
    wv = wave_extension.get_waveforms_one_unit(unit_id=unit_id)

    # get the site ids (sparse)
    c_ids = wave_extension.sparsity.unit_id_to_channel_ids[unit_id]

    # get the nearest site
    max_chid = max_chids[unit_id]
    max_chid_ix = np.where(c_ids == max_chid)[0][0]
    
    # get the average waveform on each site
    mean_spikes = wv.mean(axis=0)
    max_spike = mean_spikes[:, max_chid_ix]
    
    # measure each site's average waveform's cosine similarity 
    # to the average max site waveform (ratio b/w 0 and 1)
    n_sites = mean_spikes.shape[1]
    cosim_weights = []
    for s_i in range(n_sites):
        cosim_weights.append(
            np.dot(max_spike, mean_spikes[:, s_i])
            / (lalgnorm(max_spike) * lalgnorm(mean_spikes[:, s_i]))
        )
    cosim_weights = np.array(cosim_weights)

    # threshold the similarity metric to be positive
    # we only look at similarity (not inverse similarity (<0))
    cosim_weights[cosim_weights < 0] = 0

    # measure the euclidean distance of the site to the nearest site
    channel_coord = channel_coord[np.isin(channel_ids, c_ids), :]
    max_chids_coord = channel_coord[max_chid_ix, :]
        
    dist = []
    for ix, _ in enumerate(c_ids):
        dist.append(_euclidean_distance(max_chids_coord, channel_coord[ix]))
    dist = np.array(dist)

    # return spatial spread
    return {
        "spatial_spread": np.dot(cosim_weights, dist), # similarity-weighted spatial distance
        "channel_distance": dist,
        "weights": cosim_weights,
    }


# QUALITY METRICS / SPIKE FEATURES FOR ALL UNITS


def get_cell_to_site_distance(Recording, Sorting):
    site_pos = Recording.get_property('location3D')
    cell_pos = np.array([Sorting.get_property('x'), Sorting.get_property('y'), Sorting.get_property('z')]).T    
    return cdist(cell_pos, site_pos).min(axis=1)


def get_cell_to_cell_distances(ground_truth):
    """Calculates the minimum cell-to-cell distances 
    based on soma location or x, y, z coordinates.

    Args:
        ground_truth: Ground truth extractor containing cell properties.

    Returns:
        numpy.ndarray: Minimum cell-to-cell distances or pairwise distances.
    """
    # if ground truth has a soma location property
    if 'soma_location' in ground_truth.get_property_keys():
        cell_loc = ground_truth.get_property('soma_location')
        cell_distance = cdist(cell_loc, cell_loc, metric='euclidean')
    elif 'x' in ground_truth.get_property_keys():
        # if ground truth has a x,y,z coordinates
        x = ground_truth.get_property('x')
        y = ground_truth.get_property('y')
        z = ground_truth.get_property('z')
        cell_pos = np.array([x, y, z]).T        
        cell_distance = cdist(cell_pos, cell_pos, metric='euclidean')
    else: 
        raise ValueError("""ground_truth should have a x, y, z coordinates or be in a 'soma_location' property""")
    
    # return distances
    np.fill_diagonal(cell_distance, np.nan)
    return np.nanmin(cell_distance, axis=0)


def get_firing_rates(sorting, duration_s):
    """
    Plots the firing rate distribution of units in a sorting extractor.

    Parameters:
    - sorting: spikeinterface SortingExtractor
        The sorting extractor containing spike trains.
    - duration_s: the total duration
    - bins: int, optional
        Number of bins for the histogram (default is 20).
    """
    unit_ids = sorting.get_unit_ids()
    firing_rates = [
        len(sorting.get_unit_spike_train(unit_id)) / duration_s
        for unit_id in unit_ids
    ]
    return firing_rates


def get_firing_ranges(analyzer):

    firing_range = sqm.compute_firing_ranges(sorting_analyzer=analyzer)
    return np.array(list(firing_range.values()))


def get_waveform_amplitude_medians(analyzer):
    amplitude_medians = sqm.compute_amplitude_medians(sorting_analyzer=analyzer)
    return np.array(list(amplitude_medians.values()))


def get_waveform_snr(analyzer):
    snrs = sqm.compute_snrs(sorting_analyzer=analyzer)
    return np.array(list(snrs.values()))


def get_spatial_spread_all_units(
    recording, analyzer, peak_sign="neg"
):
    """get all units' spatial extent metrics

    Args:
        recording_path (str): Path of the Recording Extractor
        study_path (str): _description_
        ms_before (float): _description_
        ms_after (float): _description_

    Returns:
        (dict): spatial spread of each unit
        - key: true unit id
        - value: spatial spread
    
    Prerequisites:
        - have extracted ground truth waveforms
    """    
    # get sites' distance to the max site
    # get 3D coordinates
    channel_coord = recording.get_probe().contact_positions
    channel_ids = recording.get_channel_ids()

    # get channels where spike amplitude is maximal
    max_chids = ttools.get_template_extremum_channel(analyzer, peak_sign=peak_sign)

    # takes 1:30 min (1310 units)
    spatial_spread = dict()
    for _, unit in enumerate(analyzer.unit_ids):
        try:
            spatial_spread[unit] = get_spatial_spread(
                analyzer, unit, max_chids, channel_ids, channel_coord
            )["spatial_spread"]
        except IndexError:
            # silent units have no waveforms (their template is zero and the site with the largest amplitude is not in their sparsity)
            spatial_spread[unit] = np.nan
    return np.array(list(spatial_spread.values()))


def get_mad_ratios(analyzer):
    """get mad ratio per unit. Values close to 1 indicate a good isolation of the unit.
    i.e., the mean absolute deviation of the spike amplitude is similar to the mean absolute deviation 
    of the noise amplitude.

    Args:
        WeOriginal (WaveformExtractor): _description_

    Returns:
        np.array: mad ratio per unit
    """
    # pre-compute negative spike amplitudes
    amplitudes_by_unit = get_waveform_amplitudes_per_unit(analyzer)

    # get nearest channels
    max_chids = ttools.get_template_extremum_channel(analyzer, peak_sign="both")

    # compute mad ratio per unit
    mad_ratio = []
    for unit in analyzer.unit_ids:
        try:
            noise_amp = get_best_site_mad_noise(analyzer, max_chids, unit)
            mad_ratio.append(get_mad_ratio(amplitudes_by_unit[unit], noise_amp))
        except IndexError:
            # silent units have no waveforms (their template is zero and the site with the largest amplitude is not in their sparsity)
            mad_ratio.append(np.nan)
    return np.array(mad_ratio)


def get_spatial_metrics(analyzer):
    return analyzer.compute(input="template_metrics", include_multi_channel_metrics=True).get_data()


def compute_features(recording, ground_truth, analyzer):
    """compute macroscopic spiking and waveform features

    Args:
        recording (_type_): _description_
        ground_truth (_type_): _description_
        analyzer (_type_): _description_

    Returns:
        _type_: _description_
    """

    # calculate waveform spatial features
    spatial_metrics = get_spatial_metrics(analyzer)

    return {
        'cell_to_site_distance': get_cell_to_site_distance(recording, ground_truth),  # cell-to-electrode distance
        'cell_distance': get_cell_to_cell_distances(ground_truth),  # cell-to-cell distance
        'firing_rates': get_firing_rates(ground_truth, duration_s=600),  # firing rate
        'firing_ranges': get_firing_ranges(analyzer),  # firing range
        'snrs': get_waveform_snr(analyzer),  # cell waveform SNR
        'amplitudes': get_waveform_amplitude_medians(analyzer),  # cell waveform amplitude
        'mad_ratios': get_mad_ratios(analyzer),  # cell waveform MAD ratio
        'spreads_simi_weighted_distance': get_spatial_spread_all_units(recording, analyzer, peak_sign="neg"),  # cell waveform spatial spread (custom)
        'spread': spatial_metrics['velocity_above'],  # spatial metric: velocity above
        'repolarization_slope': spatial_metrics['repolarization_slope'],  # repolarization slope
        'exp_decay': spatial_metrics['exp_decay'],  # exponential decay
        'peak_to_valley': spatial_metrics['peak_to_valley'],  # peak to valley duration
        'velocity_above': spatial_metrics['velocity_above'],  # spatial metric: velocity above
        'velocity_below': spatial_metrics['velocity_below']  # spatial metric: velocity below
    }

# PLOTS
