import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


def conv1D_with_padding(spike: pd.Series, template: np.array):
    """1D-convolve spike snippet with template

    Args:
        spike (pd.Series): spike snippet
        template (np.array): universal template

    Returns:
        dict: _description_
    """
    # pad spike for convolution
    template_size = len(template)
    pad = np.zeros(template_size)
    padded_spike = np.hstack([pad, spike, pad])

    # convolve
    conved_signal = []
    for ix in range(2 * template_size + 1):
        amplitude_x = np.dot(template, padded_spike[ix : ix + template_size])
        conved_signal.append(amplitude_x)

    # get template shift that maximizes explained variance
    temp_shift = np.argmax(conved_signal)

    # best fitted template
    best_fit_amplit = conved_signal[temp_shift]
    best_fit_temp = best_fit_amplit * template

    # spike timestamp determined by deconvolution (at the center of the aligned template)
    spike_timestamp = int(temp_shift + len(template) / 2)
    return {
        "conved_signal": conved_signal,
        "best_fit_temp": best_fit_temp,
        "temp_shift": temp_shift,
        "detected_spike_timestamp": spike_timestamp,
    }


# def conv1D(spike: pd.Series, template: np.array):
#     """1D-convolve spike snippet with template

#     Args:
#         spike (pd.Series): spike snippet
#         template (np.array): universal template

#     Returns:
#         dict: _description_
#     """
#     # pad spike for convolution
#     template_size = len(template)

#     # cast spike as array
#     spike = np.array(spike)

#     # convolve
#     conved_signal = []
#     for ix in range(2 * template_size + 1):
#         amplitude_x = np.dot(template, spike[ix : ix + template_size])
#         conved_signal.append(amplitude_x)

#     # get template shift that maximizes explained variance
#     temp_shift = np.argmax(conved_signal)

#     # best fitted template
#     best_fit_amplit = conved_signal[temp_shift]
#     best_fit_temp = best_fit_amplit * template

#     # spike timestamp determined by deconvolution (at the center of the aligned template)
#     spike_timestamp = int(temp_shift + len(template) / 2)
#     return {
#         "conved_signal": conved_signal,
#         "best_fit_temp": best_fit_temp,
#         "temp_shift": temp_shift,
#         "detected_spike_timestamp": spike_timestamp,
#     }


def conv1D(spike: pd.Series, template: np.array):
    """1D-convolve spike snippet onto
    implemented with the cross-correlation formulation

    This is the same as the cross-correlation of spike and template
    and can possibly be implemented faster with the convolution
    np.convolve(np.flip(template), spike)

    Args:
        spike (pd.Series): spike snippet
        template (np.array): universal template

    Returns:
        np.array: convolved signal
    """

    # count template timepoints
    template_size = len(template)

    # cast spike as array
    spike = np.array(spike)

    # convolve
    conved_signal = []
    for ix in range(2 * template_size + 1):
        amplitude_x = np.dot(template, spike[ix : ix + template_size])
        conved_signal.append(amplitude_x)
    return conved_signal


def conv1D_pachitariu(spike: pd.Series, template: np.array):
    """1D-convolve spike snippet onto
    implemented with the cross-correlation formulation

    This is the same as the cross-correlation of spike and template
    and can possibly be implemented faster with the convolution
    np.convolve(np.flip(template), spike)

    Args:
        spike (pd.Series): spike snippet
        template (np.array): universal template

    Returns:
        np.array: convolved signal
    """

    # count template timepoints
    template = template.reshape(len(template), 1)
    template_size = len(template)

    # cast spike as array
    spike = np.array(spike)
    spike = spike.reshape(len(spike), 1)

    # convolve
    conved_signal = []
    for ix in range(2 * template_size + 1):
        amplitude_x = float(
            np.transpose(template) @ spike[ix : ix + template_size, :]
        )
        conved_signal.append(amplitude_x)
    return conved_signal


def get_best_fit_template(conved_signal, template, best_fit_shift):
    best_fit_amplit = conved_signal[best_fit_shift]
    best_fit_temp = best_fit_amplit * template
    return best_fit_temp


def set_timestamp_near_true_timestamp(
    conved_signal, neighborhood: tuple, template_len: int, sampling_freq: int
):
    """constrain the position of the detected spike timestamp within a restricted
    neighborhood of the true timestamp to ensure we do not detect another spike
    or noise

    Args:
        conved_signal (_type_): _description_
        neighborhood (tuple): _description_
        template_len (int): _description_
        sampling_freq (int): _description_

    Returns:
        _type_: _description_
    """
    # get true timestamp neighborhood start and end
    # we constrain the best match to be in
    start_tpoints = int(neighborhood[0] * sampling_freq / 1000)
    end_tpoints = int(neighborhood[1] * sampling_freq / 1000)

    # calculate best fit shift near the spike true timestamp
    near_conved_signal = conved_signal[start_tpoints:end_tpoints]
    best_fit_shift = start_tpoints + np.argmax(near_conved_signal)

    # get timestamp timepoint location
    spike_timestamp = int(best_fit_shift + template_len / 2)
    return {
        "spike_timestamp": spike_timestamp,
        "best_fit_shift": best_fit_shift,
    }


def template_match(
    spikes: pd.DataFrame,
    template: np.array,
    neighborhood: tuple,
    timepoints_before: int,
    timepoints_after: int,
    sampling_freq: int,
    figsize: tuple,
):
    """apply template matching to spike snippets, constraining
    peak detection within a restricted neighborhood of the
    ground truth spike timestamps

    Args:
        spikes (pd.DataFrame): _description_
        template (np.array): _description_
        neighborhood (tuple): typically (6,9) ms after the start of the spike snippet
        timepoints_before: int,
        timepoints_after: int,
        sampling_freq: int,
        figsize: tuple,

    Returns:
        dict:
       - np.array of detected spikes as signal that best match template
       - np.array of the scaled templates that best fitted each spike
    """

    # initialize
    best_fit_temps = []
    detected_spikes = []
    max_explained_var = []
    best_fit_shifts = []

    # count spikes
    n_spikes = spikes.shape[0]

    # setup plot
    fig, axes = plt.subplots(n_spikes, 3, figsize=figsize)

    # loop over spike snippets
    for spike_i in range(n_spikes):
        # convolve spike snippet with template
        spike = spikes.iloc[spike_i]
        conved_signal = conv1D_pachitariu(spike, template)

        # constrain timestamp timepoint location within true timestamp neighborhood
        # to avoid detecting another nearby spike
        ttp_out = set_timestamp_near_true_timestamp(
            conved_signal=conved_signal,
            template_len=len(template),
            neighborhood=neighborhood,
            sampling_freq=sampling_freq,
        )

        # get best fit template
        best_fit_temp = get_best_fit_template(
            conved_signal, template, ttp_out["best_fit_shift"]
        )

        # cast spike as array
        raw_spike = np.array(spike)

        # get the best-matched signal from the raw spike as the detected spike
        detected_spike = raw_spike[
            ttp_out["spike_timestamp"]
            - timepoints_before : ttp_out["spike_timestamp"]
            + timepoints_after
        ]
        # get snippet time (ms)
        ms_before = timepoints_before / sampling_freq * 1000
        ms_after = timepoints_after / sampling_freq * 1000
        n_timepoints = timepoints_before + timepoints_after
        timesteps = [-ms_before, 0, ms_after]
        xticks = [0, n_timepoints / 2, n_timepoints]

        # make axis robust to edge case of 1 spike
        import copy

        axes_ = copy.copy(axes)
        if n_spikes > 1:
            axes_ = copy.copy(axes[0, :])
            axes_[0] = axes[spike_i, 0]
            axes_[1] = axes[spike_i, 1]
            axes_[2] = axes[spike_i, 2]

        axes_[0].plot(detected_spike)
        axes_[0].plot(best_fit_temp, "r")

        # add legend
        axes_[0].set_xticks(xticks)
        axes_[0].set_xticklabels(timesteps)
        if spike_i == n_spikes - 1:
            axes_[0].set_xlabel("time (ms)")

        # plot raw spike with template
        axes_[1].plot(raw_spike)
        axes_[1].plot(template, "r")

        # add legend
        if spike_i == 0:
            axes_[1].set_title("raw spike snippet", fontsize=9)
        if spike_i == n_spikes - 1:
            axes_[1].set_xlabel("timepoints", fontsize=9)

        # calculate explained variance by template shift
        # explained_var = (
        #     np.array(conved_signal) ** 2 / np.linalg.norm(spike, ord=2) ** 2
        # )

        # test Pachitariu explained variance
        explained_var = np.array(conved_signal) ** 2

        # plot explained variances
        axes_[2].plot(explained_var)

        # add legend
        if spike_i == 0:
            axes_[2].set_title(
                "explained variance by shift (ratio)", fontsize=9
            )
            axes_[2].set_ylabel("explained variance (ratio)", fontsize=9)
        if spike_i == n_spikes - 1:
            axes_[2].set_xlabel("timepoints", fontsize=9)

        # record best fit template and best-matched signal (detected spike)
        best_fit_temps.append(best_fit_temp)
        detected_spikes.append(detected_spike)
        max_explained_var.append(np.max(explained_var))
        best_fit_shifts.append(ttp_out["best_fit_shift"])

    # cast as arrays
    detected_spikes = np.array(detected_spikes)
    best_fit_temps = np.array(best_fit_temps)
    max_explained_var = np.array(max_explained_var)

    plt.tight_layout()
    return {
        "detected_spikes": detected_spikes,
        "best_fit_temps": best_fit_temps,
        "max_explained_var": max_explained_var,
        "best_fit_shifts": best_fit_shifts,
        "fig": fig,
    }


def template_match_with_padding(spikes: pd.DataFrame, template: np.array):
    """apply template matching to spike snippets padded with zeros at
    both sides

    Args:
        spikes (pd.DataFrame): _description_
        template (np.array): _description_

    Returns:
        dict:
       - np.array of detected spikes as signal that best match template
       - np.array of the scaled templates that best fitted each spike
    """
    # initialize
    best_fit_temps = []
    detected_spikes = []

    # count spikes
    n_spikes = spikes.shape[0]

    # plot
    fig, axes = plt.subplots(n_spikes, 3, figsize=(13, 10))

    # loop over spike snippets
    for spike_i in range(n_spikes):
        # convolve spike snippet with template
        spike = spikes.iloc[spike_i]
        (
            conved_signal,
            best_fit_temp,
            _,
            detected_spike_timestamp,
        ) = conv1D_with_padding(spike, template).values()

        # padd spike from -9 to + 9 (adding zero pads)
        template_size = len(template)
        pad = np.zeros(template_size)
        raw_spike = np.hstack([pad, spike, pad])

        # get the best-matched signal from the raw spike as the detected spike
        detected_spike = raw_spike[
            detected_spike_timestamp - 30 : detected_spike_timestamp + 30
        ]

        # plot raw spike with template
        axes[spike_i, 0].plot(raw_spike)
        axes[spike_i, 0].plot(template, "r")
        axes[spike_i, 0].set_title("raw spike (-9 to + 9 ms)")

        # plot aligned spike with template
        axes[spike_i, 1].plot(detected_spike)
        axes[spike_i, 1].plot(best_fit_temp, "r")
        axes[spike_i, 1].set_title("aligned (detected) spike (-3 to + 3 ms)")
        axes[spike_i, 1].set_xlabel("timepoints")

        # plot explained variance by template shift
        axes[spike_i, 2].plot(np.array(conved_signal) ** 2)
        axes[spike_i, 2].set_ylabel("explained variance (W^T.D)**2")
        axes[spike_i, 2].set_title("explained variance by shift")

        # record best fit template and best-matched signal (detected spike)
        best_fit_temps.append(best_fit_temp)
        detected_spikes.append(detected_spike)

    # cast as arrays
    detected_spikes = np.array(detected_spikes)
    best_fit_temps = np.array(best_fit_temps)

    plt.tight_layout()
    return {
        "detected_spikes": detected_spikes,
        "best_fit_temps": best_fit_temps,
        "fig": fig,
    }


def get_rsquared(instances: np.ndarray, template_fits: np.ndarray):
    """calculate r-squared (proportion of explained variance) of
    template fits to spikes

    Args:
        instances (np.ndarray): _description_
        template_fits (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    rsquared = []
    for ix in range(len(instances)):
        SSE = sum((instances[ix, :] - template_fits[ix, :]) ** 2)
        SST = sum((instances[ix, :] - instances[ix, :].mean()) ** 2)
        rsquared.append(1 - SSE / SST)
    return rsquared
