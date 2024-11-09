import numpy as np
from matplotlib import pyplot as plt

def plot_exc_spikes(
    traces, time, exc, exc_mtypes, order, ttp_sample, spike_start, spike_end, before_ms, after_ms, n_rows, n_cols, fig_size, tight_layout_cfg
):

    # re-order the cells
    # order first positive peak larger than second positive peak first

    assert len(order) == len(exc), "wrong order length"
    exc_ordered1 = [exc[o_i] for o_i in order[:n_rows]]
    exc_ordered2 = [exc[o_i] for o_i in order[n_rows:]]

    # plot
    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size)

    # first column of cells
    for e_i, traces in enumerate(exc_ordered1):

        time = traces[0].index - np.min(traces[0].index)

        # electrode above
        ax[e_i, 0].plot(
            time[spike_start:spike_end],
            traces[0].values[spike_start:spike_end],
            c="tab:red",
        )
        # electrode below
        ax[e_i, 0].plot(
            time[spike_start:spike_end],
            traces[1].values[spike_start:spike_end],
            c="tab:orange",
        )

        # axes
        ax[e_i, 0].spines[["top", "right"]].set_visible(False)
        ax[e_i, 0].legend("", frameon=False)
        if not e_i == len(exc_ordered1) - 1:
            ax[e_i, 0].set_xticks([])
        else:
            ax[e_i, 0].set_xticks(
                [time[spike_start], time[ttp_sample], time[spike_end]],
                [-before_ms, 0, after_ms],
            )
            ax[e_i, 0].set_xlabel("Time (ms)")

        # disconnect axes (R style)
        ax[e_i, 0].spines["bottom"].set_position(("axes", -0.05))
        ax[e_i, 0].yaxis.set_ticks_position("left")
        ax[e_i, 0].spines["left"].set_position(("axes", -0.05))

        # annotate
        ax[e_i, 0].set_title(exc_mtypes[order[:n_rows][e_i]])

    # second column of cells
    for e_i, traces in enumerate(exc_ordered2):

        time = traces[0].index - np.min(traces[0].index)

        ax[e_i, 1].plot(
            time[spike_start:spike_end],
            traces[0].values[spike_start:spike_end],
            c="tab:red",
        )
        ax[e_i, 1].plot(
            time[spike_start:spike_end],
            traces[1].values[spike_start:spike_end],
            c="tab:orange",
        )
        ax[e_i, 1].set_title(order[e_i])

        # axes
        ax[e_i, 1].spines[["top", "right"]].set_visible(False)
        ax[e_i, 1].legend("", frameon=False)
        if not e_i == len(exc_ordered2) - 1:
            ax[e_i, 1].set_xticks([])
        else:
            ax[e_i, 1].set_xticks(
                [time[spike_start], time[ttp_sample], time[spike_end]],
                [-before_ms, 0, after_ms],
            )
            ax[e_i, 1].set_xlabel("Time (ms)")

        # disconnect axes (R style)
        ax[e_i, 1].spines["bottom"].set_position(("axes", -0.05))
        ax[e_i, 1].yaxis.set_ticks_position("left")
        ax[e_i, 1].spines["left"].set_position(("axes", -0.05))

        # annotate
        ax[e_i, 1].set_title(exc_mtypes[order[n_rows:][e_i]])

    fig.tight_layout(**tight_layout_cfg)
    return fig


def plot_inh_spikes(
    traces, time, inh, inh_mtypes, order, ttp_sample, spike_start, spike_end, before_ms, after_ms, n_rows, n_cols, fig_size, tight_layout_cfg
):

    # re-order the cells
    # order first positive peak larger than second positive peak first

    assert len(order) == len(inh), "wrong order length"
    inh_ordered1 = [inh[o_i] for o_i in order[:n_rows]]
    inh_ordered2 = [inh[o_i] for o_i in order[n_rows:]]

    # plot
    fig, ax = plt.subplots(n_rows, n_cols, figsize=fig_size)

    # first column of cells
    for e_i, traces in enumerate(inh_ordered1):

        time = traces[0].index - np.min(traces[0].index)

        # electrode above
        ax[e_i, 0].plot(
            time[spike_start:spike_end],
            traces[0].values[spike_start:spike_end],
            c="tab:blue",
        )
        # electrode below
        ax[e_i, 0].plot(
            time[spike_start:spike_end],
            traces[1].values[spike_start:spike_end],
            c="tab:purple",
        )

        # axes
        ax[e_i, 0].spines[["top", "right"]].set_visible(False)
        ax[e_i, 0].legend("", frameon=False)
        if not e_i == len(inh_ordered1) - 1:
            ax[e_i, 0].set_xticks([])
        else:
            ax[e_i, 0].set_xticks(
                [time[spike_start], time[ttp_sample], time[spike_end]],
                [-before_ms, 0, after_ms],
            )
            ax[e_i, 0].set_xlabel("Time (ms)")

        # disconnect axes (R style)
        ax[e_i, 0].spines["bottom"].set_position(("axes", -0.05))
        ax[e_i, 0].yaxis.set_ticks_position("left")
        ax[e_i, 0].spines["left"].set_position(("axes", -0.05))

        # annotate
        ax[e_i, 0].set_title(inh_mtypes[order[:n_rows][e_i]])

    # second column of cells
    for e_i, traces in enumerate(inh_ordered2):

        time = traces[0].index - np.min(traces[0].index)

        ax[e_i, 1].plot(
            time[spike_start:spike_end],
            traces[0].values[spike_start:spike_end],
            c="tab:blue",
        )
        ax[e_i, 1].plot(
            time[spike_start:spike_end],
            traces[1].values[spike_start:spike_end],
            c="tab:purple",
        )
        ax[e_i, 1].set_title(order[e_i])

        # axes
        ax[e_i, 1].spines[["top", "right"]].set_visible(False)
        ax[e_i, 1].legend("", frameon=False)
        if not e_i == len(inh_ordered2) - 1:
            ax[e_i, 1].set_xticks([])
        else:
            ax[e_i, 1].set_xticks(
                [time[spike_start], time[ttp_sample], time[spike_end]],
                [-before_ms, 0, after_ms],
            )
            ax[e_i, 1].set_xlabel("Time (ms)")

        # disconnect axes (R style)
        ax[e_i, 1].spines["bottom"].set_position(("axes", -0.05))
        ax[e_i, 1].yaxis.set_ticks_position("left")
        ax[e_i, 1].spines["left"].set_position(("axes", -0.05))

        # annotate
        ax[e_i, 1].set_title(inh_mtypes[order[n_rows:][e_i]])

    fig.tight_layout(**tight_layout_cfg)
    return fig