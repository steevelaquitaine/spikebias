"""plot utils for models
"""


def locate_spike_by_mt(we, job_kwargs: dict):
    """locate spikes
    - with monopolar_triangulation
    - both positive and negative peaks

    Returns:
        [('x', 'float64'),  ('y', 'float64'), ('z', 'float64'), ('alpha', 'float64')]
    """

    spike_loc = si.postprocessing.compute_spike_locations(
        we,
        load_if_exists=False,
        ms_before=0.5,
        ms_after=0.5,
        spike_retriver_kwargs={
            "channel_from_template": True,
            "peak_sign": "both",
            "radius_um": 50,
        },
        method="monopolar_triangulation",
        method_kwargs={},
        outputs="by_unit",
        **job_kwargs,
    )
    return spike_loc[0]


def plot_waveforms_2D(ax, we, unit_id, n_sites, site_coord):
    """plot first 2D waveform, all 1D waveform
    on nearest site

    Args:
        ax (_type_): _description_
        we (_type_): _description_
        unit_id (_type_): _description_

    Returns:
        _type_: _description_
    """

    # get all units nearest channels (with extremum amplitude)
    max_chids = ttools.get_template_extremum_channel(we, peak_sign="both")

    # get its waveforms (num_spikes, num_samples, num_channels)
    wv = we.get_waveforms(unit_id=unit_id)

    # get the nearest channel
    c_ids = we.sparsity.unit_id_to_channel_ids[unit_id]
    max_chid = max_chids[unit_id]
    max_chid_ix = np.where(c_ids == max_chid)[0][0]

    # zoom in on n nearest sites
    c_ids, c_ix = get_n_nearest_sites(site_coord, max_chid, c_ids, n_sites)

    # plot
    ax[0].imshow(wv[0, :, c_ix], cmap="viridis", aspect="auto")
    ax[0].set_title(f"unit {unit_id}")
    ax[0].set_yticks(np.arange(0, len(c_ids), 2), c_ids[::2])
    ax[1].plot(wv[:, :, max_chid_ix].T)
    return ax


def plot_2d_1d_waveform_and_sites(we, unit_id, n_site, site_coord):

    # get all units nearest channels (with extremum amplitude)
    max_chids = ttools.get_template_extremum_channel(we, peak_sign="both")

    # get the nearest channel
    c_ids = we.sparsity.unit_id_to_channel_ids[unit_id]
    max_chid = max_chids[unit_id]
    print("channels:", c_ids)
    print("nearest channel:", max_chid)

    # plot site layout
    site_c = site_coord[c_ids, :]
    ax = plt.plot(site_c[:, 0], site_c[:, 1], "o")
    for ix in range(len(site_c)):
        plt.annotate(text=c_ids[ix], xy=(site_c[ix, 0], site_c[ix, 1]))

    # plot 2D waveform
    _, ax = plt.subplots(1, 2, figsize=(5, 2))
    ax = plot_waveforms_2D(ax, we, unit_id, n_site, site_coord)
    return ax


def plot_sites_xy(ax, Probe):
    """ """

    ax.scatter(
        Probe.contact_positions[:, 0],
        Probe.contact_positions[:, 1],
        marker="o",
        color="k",
        facecolors="none",
        s=1,
    )
    ax.set_xlabel("x: width", fontsize=20)
    ax.set_ylabel("y: depth", fontsize=20)
    return ax


def plot_spike_quality(ax, spike_loc, unit, title, quality, color_tp, color_fp):
    spike_c = spike_loc[unit]
    for sp_i in range(len(spike_c)):
        if quality[sp_i] == "TP":
            color = color_tp
        else:
            color = color_fp
        ax.scatter(spike_c[sp_i][0], spike_c[sp_i][1], color=color, s=1)
    ax.set_title(title)


def plot_waveforms(ax, we, max_chids, unit, before_ms, after_ms):

    # get waveforms
    wv, spike_ix = we.get_waveforms(unit_id=unit, with_index=True)

    # get channel ids (sparse)
    c_ids = we.sparsity.unit_id_to_channel_ids[unit]

    # get nearest channel
    max_chid = max_chids[unit]
    max_chid_ix = np.where(c_ids == max_chid)[0][0]

    # plot waveforms (num_spikes, num_samples, num_channels)
    ax.plot(wv[:5, :, max_chid_ix].T)
    ax.set_xticks([0, wv.shape[1], wv.shape[1]], [before_ms, 0, after_ms])
    ax.spines["right"].set_visible(False)
    return (ax, spike_ix, c_ids, max_chid)


def plot_all_unit_waveforms(
    we, comp, max_chids, Probe, unit_ids, spike_loc_mt, n_spikes, metrics
):
    # loop over units and plot waveforms
    for _, unit in enumerate(unit_ids):

        fig = plt.figure(figsize=(6, 3))

        # plot
        ax = fig.add_subplot(121)

        # PANEL 1: spike location
        quality = comp.get_labels2(unit)[0]
        ax = plot_sites_xy(ax, Probe)
        plot_spike_quality(ax, spike_loc_mt, unit, "good", quality, "r", [0, 0.6, 1])

        # aesthetics
        ax.set_xlim([3500, 4600])
        ax.set_ylim([-1500, -1100])
        ax.spines["right"].set_visible(False)

        # report
        print(f"unit {unit}")
        print("best score:", sum(quality == "TP") / len(quality))
        print("firing rate:", n_spikes[unit] / 600)
        print("qmetrics:\n", metrics.iloc[unit])

        # PANEL 2: plot waveforms
        ax = fig.add_subplot(122)
        ax = plot_waveforms(ax, we, max_chids, unit, -3, 3)


def plot_2d_waveform_on_neareast_sites(ax, we, site_coord, n_sites: int, unit_id: int):
    """plot 2d waveform on the sites nearest to the unit

    Args:
        we (_type_): _description_
        unit (_type_): _description_
        n_sites (int, optional): _description_. Defaults to 10.

    Returns:
        _type_: _description_
    """
    # get all units nearest channels (with extremum amplitude)
    max_chids = ttools.get_template_extremum_channel(we, peak_sign="both")

    # get its waveforms (num_spikes, num_samples, num_channels)
    wv = we.get_waveforms(unit_id=unit_id)

    # get its nearest channel
    c_ids = we.sparsity.unit_id_to_channel_ids[unit_id]
    max_chid = max_chids[unit_id]

    # drop sites away from nearest site to get the
    # same number of sites for all waveforms
    c_ids_new, ix_new = get_n_nearest_sites(site_coord, max_chid, c_ids, n_sites)

    # plot
    ax[0].imshow(wv[0, :, ix_new], cmap="viridis", aspect="auto")
    ax[0].set_yticklabels(c_ids_new)
    ax[1].imshow(wv[0, :, :].T, cmap="viridis", aspect="auto")
    return ax