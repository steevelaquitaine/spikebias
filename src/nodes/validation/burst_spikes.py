"""Spike waveforms within bursts of spikes (Supplementary Fig. 3c-f)

Functions copied from notebooks/1_results/supp_fig3_CDEF_burst_modulation_of_spike_revised.ipynb.
"""

import numpy as np
from scipy.spatial.distance import cdist


def shorten_waveform(waveforms: np.array, window_to_search_peak_sec=2.5, sampling_freq=20000):

    # calculate shorter window
    sample_sec = (1 / sampling_freq) * 1000
    shorter_window = int(window_to_search_peak_sec / sample_sec)

    # get shorter window
    mid_point = int(waveforms.shape[1] / 2)
    wind_start = int(mid_point - shorter_window / 2)
    wind_end = int(mid_point + shorter_window / 2)
    shorter_window_sample = np.arange(wind_start, wind_end, 1).astype(int)
    return waveforms[:, shorter_window_sample], wind_start, wind_end


def align_by_neg_peak_and_crop(waves, sampling_frequency, window_to_search_peak_sec=2.5, window=100, peak_pos=50):
    """
    waves: array of shape (n_waveforms, n_samples) e.g. (N, 480)
    window: number of samples in the output window (e.g. 100)
    peak_pos: index in [0, window-1] where the negative peak will be placed (e.g. 50)
    Returns: array of shape (n_waveforms, window), nan where the window runs beyond the edges
    """

    # find peak index in shortened waveform
    # shorten waveform to find peak without spike collision
    short_waves, wind_start, wind_end = shorten_waveform(waves, window_to_search_peak_sec=window_to_search_peak_sec, sampling_freq=sampling_frequency)

    num_waves, _ = short_waves.shape
    left = peak_pos
    right = window - left  # number of samples to include to the right of the peak
    aligned = np.full((num_waves, window), np.nan, dtype=short_waves.dtype)  # placeholder of nans

    # index of negative peak for each waveform
    trough_idx = np.argmin(short_waves, axis=1)

    # locate peak index in longer waveforms
    trough_idx = trough_idx + wind_start

    # number of timepoints in longer waves
    n_timepoints = waves.shape[1]

    for w_ix in range(num_waves):

        # desired window in original signal: [p-left, p+right)
        src_start = max(0, trough_idx[w_ix] - left)
        src_end = min(n_timepoints, trough_idx[w_ix] + right)

        # where to place it in the output (shift if we clipped at edges)
        dst_start = max(0, left - trough_idx[w_ix])  # how many samples we clipped on the left
        dst_end = dst_start + (src_end - src_start)

        # align
        aligned[w_ix, dst_start:dst_end] = waves[w_ix, src_start:src_end]
    return aligned


def find_bursts_isi(spike_times, isi_threshold=0.01, min_spikes=3, min_duration=0.02):
    """Find bursts based on inter-spike interval (ISI) criterion.

    Parameters:
    spike_times: array-like, spike timestamps in seconds
    isi_threshold: float, maximum ISI within burst (seconds)
    min_spikes: int, minimum number of spikes to constitute a burst
    min_duration: float, minimum burst duration (seconds)

    Returns:
    list of tuples: [(burst_start, burst_end, n_spikes, burst_duration), ...]
    """
    if len(spike_times) < min_spikes:
        return []

    spike_times = np.array(spike_times)
    spike_times = np.sort(spike_times)  # ensure sorted

    # Calculate inter-spike intervals
    isis = np.diff(spike_times)

    # Spikes are in same burst if ISI < threshold
    in_burst = (0 <= isis) & (isis <= isi_threshold)

    # find burst start and end indices
    burst_starts = []
    burst_ends = []

    i = 0
    while i < len(in_burst):
        if in_burst[i]:  # Start of potential burst
            burst_start_idx = i

            # Find end of burst
            while i < len(in_burst) and in_burst[i]:
                i += 1
            burst_end_idx = i  # One past the last spike in burst

            # Check if this burst meets criteria
            n_spikes_in_burst = burst_end_idx - burst_start_idx + 1
            burst_duration = spike_times[burst_end_idx] - spike_times[burst_start_idx]

            if n_spikes_in_burst >= min_spikes and burst_duration >= min_duration:
                burst_starts.append(burst_start_idx)
                burst_ends.append(burst_end_idx)
        else:
            i += 1

    # extract burst information
    bursts = []
    for start_idx, end_idx in zip(burst_starts, burst_ends):
        burst_start_time = spike_times[start_idx]
        burst_end_time = spike_times[end_idx]
        n_spikes = end_idx - start_idx + 1
        duration = burst_end_time - burst_start_time

        bursts.append((burst_start_time, burst_end_time, n_spikes, duration))

    return bursts


def get_burst_first_n_waveforms(Recording, Gtruth, bursts, b_unit, sfreq, peak_pos=50, window=100, frames_before=100, frames_after=100, num_spikes=2):
    """Waveforms of the first `num_spikes` spikes of each burst of a unit, on the site nearest to the unit, aligned on their negative peak

    Returns an array (bursts, spikes, frames)
    """

    # find the unit's nearest electrode
    site_ids = Recording.get_channel_ids()
    sites_coord = Recording.get_channel_locations(axes="xyz")
    b_unit_loc = np.where(Gtruth.unit_ids == b_unit)[0][0]
    units_coord = np.array([Gtruth.get_property("x"), Gtruth.get_property("y"), Gtruth.get_property("z")]).T
    b_unit_coord = units_coord[b_unit_loc, :]
    dist_matrix = cdist(b_unit_coord[None], sites_coord)
    closest_site = np.argmin(dist_matrix)

    # get the spike frames (timestamps) of the unit
    unit_spike_frames = Gtruth.get_unit_spike_train(b_unit)

    # store the first waves of each burst
    waves_by_burst = []

    # loop over each burst and collect its first spike waveforms
    # all aligned on their negative peak at peak_pos
    for b_ix in range(len(bursts[b_unit])):
        try:
            burst = bursts[b_unit][b_ix]
            start_frame = int(burst[0] * sfreq)
            end_frame = int(burst[1] * sfreq)

            burst_frames_loc = (start_frame <= unit_spike_frames) & (end_frame >= unit_spike_frames)
            burst_frames = unit_spike_frames[burst_frames_loc]

            # store the waves in this burst
            waves = []
            for s_i, _ in enumerate(burst_frames):
                wave = Recording.get_traces(start_frame=burst_frames[s_i] - frames_before, end_frame=burst_frames[s_i] + frames_after, channel_ids=[site_ids[closest_site]])
                waves.append(wave)
            waves = np.array(waves).squeeze()

            # align waves at peak_pos, crop to window frames (waves, frames)
            aligned_waves = align_by_neg_peak_and_crop(np.array(waves), sfreq, window_to_search_peak_sec=2.5, window=window, peak_pos=peak_pos)

            # store the first waves of each burst
            if aligned_waves.shape[0] >= num_spikes:
                waves_by_burst.append(aligned_waves[:num_spikes, :])
        except Exception:
            pass

    # (burst, waves, frames)
    return np.array(waves_by_burst)


def ci95(data: np.array) -> np.array:
    """95% confidence interval of the mean along axis 0 (ignoring nan)"""
    return 1.96 * np.nanstd(data, axis=0) / np.sqrt(np.sum(~np.isnan(data), axis=0))


def find_in_isi_range(spike_times, min_isi=0.01, max_isi=0.02, min_spikes=3, min_duration=0.02):
    """Find sequences of spikes whose inter-spike intervals (ISI) are all within a range

    Parameters:
    spike_times: array-like, spike timestamps in seconds
    min_isi, max_isi: float, ISI range (seconds)
    min_spikes: int, minimum number of spikes in a sequence
    min_duration: float, minimum duration of a sequence (seconds)

    Returns:
    list of tuples: [(start, end, n_spikes, duration), ...]
    """
    if len(spike_times) < min_spikes:
        return []

    spike_times = np.array(spike_times)
    spike_times = np.sort(spike_times)  # ensure sorted

    # Spikes are in the same sequence if their ISI is within the range
    isis = np.diff(spike_times)
    in_isi_range = (min_isi <= isis) & (isis <= max_isi)

    # find sequence start and end indices
    isi_sequence_starts = []
    isi_sequence_ends = []

    i = 0
    while i < len(in_isi_range):
        if in_isi_range[i]:
            isi_sequence_start_idx = i

            # Find end of the sequence
            while i < len(in_isi_range) and in_isi_range[i]:
                i += 1
            isi_sequence_end_idx = i  # One past the last spike in the sequence

            # Check if this sequence meets criteria
            n_spikes_in_isi_range = isi_sequence_end_idx - isi_sequence_start_idx + 1
            sequence_duration = spike_times[isi_sequence_end_idx] - spike_times[isi_sequence_start_idx]

            if n_spikes_in_isi_range >= min_spikes and sequence_duration >= min_duration:
                isi_sequence_starts.append(isi_sequence_start_idx)
                isi_sequence_ends.append(isi_sequence_end_idx)
        else:
            i += 1

    # extract sequence information
    isi_sequences = []
    for start_idx, end_idx in zip(isi_sequence_starts, isi_sequence_ends):
        start_time = spike_times[start_idx]
        end_time = spike_times[end_idx]
        isi_sequences.append((start_time, end_time, end_idx - start_idx + 1, end_time - start_time))
    return isi_sequences


def trough_width_halfmax(waveform, fs):
    """Width (ms) of the trough of a waveform at half of its amplitude"""
    wf = waveform

    # Find trough (minimum)
    trough_idx = np.argmin(wf)
    trough_val = wf[trough_idx]

    # Half amplitude value
    half_val = trough_val / 2.0  # still negative

    # Find left crossing (before trough)
    left_idx_candidates = np.where(wf[:trough_idx] > half_val)[0]
    left_idx = left_idx_candidates[-1] if len(left_idx_candidates) > 0 else 0

    # Find right crossing (after trough)
    right_idx_candidates = np.where(wf[trough_idx:] > half_val)[0]
    right_idx = right_idx_candidates[0] + trough_idx if len(right_idx_candidates) > 0 else len(wf) - 1

    # Compute width in ms
    return ((right_idx - left_idx) / fs) * 1000.0


def mean_waveforms_by_isi_range(Recording, Gtruth, unit_ids, sfreq, min_isi, max_isi, min_spikes, window_frames, num_spikes, min_duration=0.02, min_amplitude=0):
    """Mean waveform of the first `num_spikes` spikes of the sequences with ISIs in a range, for the units with more than one sequence

    Units whose mean waveforms are all nan or whose trough is shallower than `min_amplitude` are dropped.

    Returns a dict {unit id: array (spikes, frames)}
    """

    # sequences of each unit
    sequences = {u: find_in_isi_range(Gtruth.get_unit_spike_train(u) / sfreq, min_isi=min_isi, max_isi=max_isi, min_spikes=min_spikes, min_duration=min_duration) for u in unit_ids}

    # mean waveforms of units with more than 1 sequence for stats
    mean_waves = {}
    for unit in unit_ids:
        if len(sequences[unit]) > 1:
            waves = get_burst_first_n_waveforms(Recording, Gtruth, sequences, unit, sfreq, peak_pos=int(window_frames / 2), window=window_frames, frames_before=100, frames_after=100, num_spikes=num_spikes)
            mean_waves[unit] = waves.mean(axis=0)

    # select high snr waveforms
    return {u: w for u, w in mean_waves.items() if w is not None and not np.isnan(w).all() and w.min() <= min_amplitude}
