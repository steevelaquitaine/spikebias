import numpy as np
import pandas as pd
from src.nodes.validation import spikestats
import copy
import seaborn as sns
from random import choices
import matplotlib
import collections 
from scipy.stats import mannwhitneyu

# set plot parameters
BOXPLOT_PMS = {
    "notch": True,
    "gap": 0.3,
    "width": 0.7,
    "linewidth": 0.5,
    "flierprops": {
            "marker": "o",
            "markerfacecolor": "w",
            "markersize": 1.5,
            "markeredgewidth": 0.5
        },
}


def plot_fr_by_layer(axes, df_nv, df_ns, df_ne, df_nb, layers, log_x_min, log_x_max, nbins, t_dec, cl):

    # figure parameters
    MARKERSIZE = 3
    EDGEWIDTH = 0.5
    X_MIN = 1e-3
    X_MAX = 1e3
    layer_loc = 0.01
    N_MJ_TCKS = 5
    N_MN_TCKS = 11

    # vivo
    y = df_nv["firing_rate"][df_nv["layer"].isin(layers)].values.astype(np.float32)
    _ = spikestats.plot_firing_rate_hist_vs_lognorm(
        y,
        log_x_min,
        log_x_max,
        nbins,
        t_dec,
        axes[0],
        label=f"{len(y)}",
        color=cl["COLOR_NV"],
        markerfacecolor=cl["COLOR_NV"],
        markersize=MARKERSIZE,
        markeredgewidth=EDGEWIDTH,
        legend=False,
        lognormal=True,
    )

    # silico spontaneous
    y = df_ns["firing_rate"][df_ns["layer"].isin(layers)].values.astype(np.float32)
    _ = spikestats.plot_firing_rate_hist_vs_lognorm(
        y,
        log_x_min,
        log_x_max,
        nbins,
        t_dec,
        axes[0],
        label=f"{len(y)}",
        color=cl["COLOR_NS"],
        markerfacecolor=cl["COLOR_NS"],
        markersize=MARKERSIZE,
        markeredgewidth=EDGEWIDTH,
        legend=False,
        lognormal=True,
    )

    # silico evoked
    y = df_ne["firing_rate"][df_ne["layer"].isin(layers)].values.astype(np.float32)
    _ = spikestats.plot_firing_rate_hist_vs_lognorm(
        y,
        log_x_min,
        log_x_max,
        nbins,
        t_dec,
        axes[0],
        label=f"{len(y)}",
        color=cl["COLOR_NE"],
        markerfacecolor=cl["COLOR_NE"],
        markersize=MARKERSIZE,
        markeredgewidth=EDGEWIDTH,
        legend=False,
        lognormal=True,
    )
    axes[0].set_xticklabels([])
    axes[0].set_xlim([X_MIN, X_MAX])
    axes[0].tick_params(axis="y")

    # annotate unit count
    axes[0].legend(
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(0.65, 1),
        handletextpad=-0.5,
        labelspacing=0,
        title="unit count"
    )
    # annotate layer
    axes[0].annotate(
        "Column",
        (layer_loc, 1),
        xycoords="axes fraction",
        fontweight="bold",
        fontsize=7,
    )

    # clear top and right, disconnect axes (R style)
    axes[0].spines[["top", "right"]].set_visible(False)
    axes[0].spines["bottom"].set_position(("axes", -0.05))
    axes[0].yaxis.set_ticks_position("left")
    axes[0].spines["left"].set_position(("axes", -0.05))

    # show logarithmic ticks    
    locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MJ_TCKS)
    locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0, 10, 1), numticks=N_MN_TCKS)
    axes[0].tick_params(which="both")
    axes[0].xaxis.set_major_locator(locmaj)
    axes[0].xaxis.set_minor_locator(locmin)
    axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    
    ### By LAYER **********************************************

    # initialize
    out_vivo_all = []
    out_sili_spont_all = []
    out_sili_ev_all = []
    n_nvs = []
    n_nss = []
    n_nes = []

    for ix in range(len(layers)):

        # vivo
        y = df_nv["firing_rate"][df_nv["layer"] == layers[ix]].values.astype(np.float32)
        out_vivo_all.append(
            spikestats.plot_firing_rate_hist_vs_lognorm(
                y,
                log_x_min,
                log_x_max,
                nbins,
                t_dec,
                axes[ix + 1],
                label=f"{len(y)}",
                color=cl["COLOR_NV"],
                markerfacecolor=cl["COLOR_NV"],
                markersize=MARKERSIZE,
                markeredgewidth=EDGEWIDTH,
                legend=False,
                lognormal=True,
            )
        )
        n_nvs.append(len(y))

        # silico spont
        y = df_ns["firing_rate"][df_ns["layer"] == layers[ix]].values.astype(np.float32)
        out_sili_spont_all.append(
            spikestats.plot_firing_rate_hist_vs_lognorm(
                y,
                log_x_min,
                log_x_max,
                nbins,
                t_dec,
                axes[ix + 1],
                label=f"{len(y)}",
                color=cl["COLOR_NS"],
                markerfacecolor=cl["COLOR_NS"],
                markersize=MARKERSIZE,
                markeredgewidth=EDGEWIDTH,
                legend=False,
                lognormal=True,
            )
        )
        n_nss.append(len(y))
        axes[ix + 1].set_xlim([X_MIN, X_MAX])

        # silico evoked
        y = df_ne["firing_rate"][df_ne["layer"] == layers[ix]].values.astype(np.float32)
        out_sili_ev_all.append(
            spikestats.plot_firing_rate_hist_vs_lognorm(
                y,
                log_x_min,
                log_x_max,
                nbins,
                t_dec,
                axes[ix + 1],
                label=f"{len(y)}",
                color=cl["COLOR_NE"],
                markerfacecolor=cl["COLOR_NE"],
                markersize=MARKERSIZE,
                markeredgewidth=EDGEWIDTH,
                legend=False,
                lognormal=True,
            )
        )
        n_nes.append(len(y))
        axes[ix + 1].set_xlim([X_MIN, X_MAX])

        if ix == 2:
            axes[ix + 1].set_ylabel("Probability (ratio)")
        if ix == 4:
            axes[ix + 1].set_xlabel("Firing rate (spikes/sec)")
        else:
            axes[ix + 1].set_xticklabels([])

        if layers[ix] == "L5":

            # add synthetic model
            y = df_nb["firing_rate"].values.astype(np.float32)
            spikestats.plot_firing_rate_hist_vs_lognorm(
                y,
                log_x_min,
                log_x_max,
                nbins,
                t_dec,
                axes[ix + 1],
                label=f"{len(y)}",
                color=cl["COLOR_NB"],
                markerfacecolor=cl["COLOR_NB"],
                markersize=MARKERSIZE,
                markeredgewidth=EDGEWIDTH,
                legend=False,
                lognormal=True,
            )
            n_nb = len(y)
            axes[ix + 1].set_xticklabels([])

        # annotate unit count
        axes[ix + 1].legend(
            frameon=False,
            loc="upper left",
            bbox_to_anchor=(0.65, 1),
            handletextpad=-0.5,
            labelspacing=0,
        )
        # annotate layer
        axes[ix + 1].annotate(
            f"{layers[ix]}",
            (layer_loc, 0.8),
            xycoords="axes fraction",
            fontweight="bold",
            fontsize=7,
        )

        # disconnected axes
        axes[ix + 1].spines[["top", "right"]].set_visible(False)
        axes[ix + 1].spines["bottom"].set_position(("axes", -0.05))
        axes[ix + 1].yaxis.set_ticks_position("left")
        axes[ix + 1].spines["left"].set_position(("axes", -0.05))

        # show logarithmic ticks    
        locmaj = matplotlib.ticker.LogLocator(base=10, numticks=N_MJ_TCKS)
        locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0,10,1), numticks=N_MN_TCKS)
        axes[ix+1].tick_params(which="both")
        axes[ix+1].xaxis.set_major_locator(locmaj)
        axes[ix+1].xaxis.set_minor_locator(locmin)
        axes[ix+1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axes[ix+1].set_xticks([0.01, 1, 100])
        
    # unit-test
    assert sum(n_nvs) == df_nv.shape[0], "total # of units is wrong"
    assert sum(n_nss) == df_ns.shape[0], "total # of units is wrong"
    assert sum(n_nes) == df_ne.shape[0], "total # of units is wrong"
    assert n_nb == df_nb.shape[0], "total # of units is wrong"
    

def plot_fr_stats_by_layer(ax, df_nv, df_ns, df_ne, df_nb, layers, cl):
    
    # concat firing rates by layer
    fr_vivo = []
    fr_sili_sp = []
    fr_sili_ev = []
    layer_vivo = []
    layer_sili_sp = []
    layer_sili_ev = []
    
    # plot each layer in a panel
    for ix, layer in enumerate(layers):

        # vivo
        fr_vivo_i = (
            df_nv["firing_rate"][df_nv["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_vivo += fr_vivo_i
        layer_vivo += [layer] * len(fr_vivo_i)

        # silico spontaneois
        fr_sili_sp_i = (
            df_ns["firing_rate"][df_ns["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_sili_sp += fr_sili_sp_i
        layer_sili_sp += [layer] * len(fr_sili_sp_i)

        # silico evoked
        fr_sili_ev_i = (
            df_ne["firing_rate"][df_ne["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_sili_ev += fr_sili_ev_i
        layer_sili_ev += [layer] * len(fr_sili_ev_i)

    # build plot dataset
    vivo_data = pd.DataFrame(data=np.array(fr_vivo), columns=["firing rate"])
    vivo_data["experiment"] = "M"
    vivo_data["layer"] = layer_vivo

    sili_data_sp = pd.DataFrame(data=np.array(fr_sili_sp), columns=["firing rate"])
    sili_data_sp["experiment"] = "NS"
    sili_data_sp["layer"] = layer_sili_sp

    sili_data_ev = pd.DataFrame(data=np.array(fr_sili_ev), columns=["firing rate"])
    sili_data_ev["experiment"] = "E"
    sili_data_ev["layer"] = layer_sili_ev

    sili_data_nb = pd.DataFrame(
        data=np.array(df_nb["firing_rate"].values.astype(np.float32)), columns=["firing rate"]
    )
    sili_data_nb["experiment"] = "S"
    sili_data_nb["layer"] = "L5"

    plot_data = pd.concat([vivo_data, sili_data_sp, sili_data_ev, sili_data_nb], ignore_index=True)

    # drop sites outside layers
    mask = np.isin(plot_data["layer"], layers)
    plot_data = plot_data[mask]
    plot_data = plot_data.sort_values(by=["layer"])

    # we plot the stats over log10(firing rate) which reflects
    # bestwhat we see from the distribution plots (stats over raw data
    # is not visible). Note: the log of the median is the median of the log
    plot_data2 = copy.copy(plot_data)
    plot_data2["firing rate"] = np.log10(plot_data2["firing rate"])

    # plot
    ax = sns.boxplot(
        ax=ax,
        data=plot_data2,
        x="layer",
        y="firing rate",
        hue="experiment",
        hue_order=["M", "NS", "E", "S"],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_NB"]],
        **BOXPLOT_PMS,
        vert=True,
    )
    # ax.set_yscale("log")
    # axes
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    # ax.set_ylabel("Log of firing rate")
    # ax.set_xlabel("Layer")

    # customize the boxplot
    p = 0
    for box in ax.patches:
        if box.__class__.__name__ == "PathPatch":
            # a list item for each layer group
            color = box.get_facecolor()                
            box.set_edgecolor(color)
            # Each box has 6 associated Line2D objects
            # (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same colour
            # as above
            # group 1 (NV)
            for k in range(6 * p, 6 * (p + 1)):
                ax.lines[k].set_color(color)  # box
                ax.lines[k].set_mfc("w")    # whisker
                ax.lines[k].set_mec(color)    # fliers
            p += 1

    # show minor ticks
    # ax.tick_params(which="both")
    # locmaj = matplotlib.ticker.LogLocator(base=10, numticks=4)
    # locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0, 10, 1), numticks=11)
    # ax.yaxis.set_major_locator(locmaj)
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return ax, plot_data2
    
    
def plot_fr_stats_by_layer_vert(ax, df_nv, df_ns, df_ne, df_nb, layers, cl):
    
    BOXPLOT_PMS = {
        "notch": True, # notches are 95% CI
        "gap": 0.2,
        "width": 0.7,
        "linewidth": 0.5,
        "flierprops": {
                "marker": "o",
                "markerfacecolor": "w",
                "markersize": 1.5,
                "markeredgewidth": 0.5
            },
        "boxprops": {"linewidth": 0.5}
    }
    alpha = 0.4
    
    # concat firing rates by layer
    fr_vivo = []
    fr_sili_sp = []
    fr_sili_ev = []
    layer_vivo = []
    layer_sili_sp = []
    layer_sili_ev = []
    
    # plot each layer in a panel
    for ix, layer in enumerate(layers):

        # vivo
        fr_vivo_i = (
            df_nv["firing_rate"][df_nv["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_vivo += fr_vivo_i
        layer_vivo += [layer] * len(fr_vivo_i)

        # silico spontaneois
        fr_sili_sp_i = (
            df_ns["firing_rate"][df_ns["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_sili_sp += fr_sili_sp_i
        layer_sili_sp += [layer] * len(fr_sili_sp_i)

        # silico evoked
        fr_sili_ev_i = (
            df_ne["firing_rate"][df_ne["layer"] == layer].values.astype(np.float32).tolist()
        )
        fr_sili_ev += fr_sili_ev_i
        layer_sili_ev += [layer] * len(fr_sili_ev_i)

    # build plot dataset
    vivo_data = pd.DataFrame(data=np.array(fr_vivo), columns=["firing rate"])
    vivo_data["experiment"] = "M"
    vivo_data["layer"] = layer_vivo

    sili_data_sp = pd.DataFrame(data=np.array(fr_sili_sp), columns=["firing rate"])
    sili_data_sp["experiment"] = "NS"
    sili_data_sp["layer"] = layer_sili_sp

    sili_data_ev = pd.DataFrame(data=np.array(fr_sili_ev), columns=["firing rate"])
    sili_data_ev["experiment"] = "E"
    sili_data_ev["layer"] = layer_sili_ev

    sili_data_nb = pd.DataFrame(
        data=np.array(df_nb["firing_rate"].values.astype(np.float32)), columns=["firing rate"]
    )
    sili_data_nb["experiment"] = "S"
    sili_data_nb["layer"] = "L5"

    plot_data = pd.concat([vivo_data, sili_data_sp, sili_data_ev, sili_data_nb], ignore_index=True)

    # drop sites outside layers
    mask = np.isin(plot_data["layer"], layers)
    plot_data = plot_data[mask]
    plot_data = plot_data.sort_values(by=["layer"])

    # we plot the stats over log10(firing rate) which reflects
    # bestwhat we see from the distribution plots (stats over raw data
    # is not visible). Note: the log of the median is the median of the log
    plot_data2 = copy.copy(plot_data)
    plot_data2["firing rate"] = np.log10(plot_data2["firing rate"])

    # plot
    ax = sns.boxplot(
        ax=ax,
        #positions=np.arange(0,15,1),
        data=plot_data2,
        x="firing rate",
        y="layer",        
        hue="experiment",
        hue_order=["M", "NS", "E", "S"],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_NB"]],
        **BOXPLOT_PMS,
        vert=False,
    )
    # axes
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.set_ylabel("Layer")
    ax.set_xlabel("Log of firing rate")

    ax = set_aes_boxes(ax, alpha)

    # show minor ticks
    # ax.tick_params(which="both")
    # locmaj = matplotlib.ticker.LogLocator(base=10, numticks=4)
    # locmin = matplotlib.ticker.LogLocator(base=10.0, subs=np.arange(0, 10, 1), numticks=11)
    # ax.yaxis.set_major_locator(locmaj)
    # ax.yaxis.set_minor_locator(locmin)
    # ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    return ax, plot_data2


def bootstrap_varlogfr(firing_rate: list, N_BOOT: int):
    """Calculate the variance of each log(firing rate) distribution
    bootstrapped from the provided "firing_rate" list.

    Args:
        firing_rate (list): _description_
        N_BOOT (int): _description_

    Returns:
        std_boot_all (list): "N_BOOT" variances. One for each 
        distribution of firing rate bootstrapped from the 
        provided "firing_rate" list
    """
    std_boot_all = []
    # - sample firing rates
    # - log transform
    # - calculate variance and repeat N_BOOT
    for ix in range(N_BOOT):
        fr_boot_i = choices(firing_rate, k=len(firing_rate))
        log_fr_boot_i = np.log10(fr_boot_i)
        std_boot_all.append(np.var(log_fr_boot_i))
    return std_boot_all


def set_aes_boxes(ax, alpha):
    """set boxplots' aesthetics

    Args:
        ax (_type_): _description_
        alpha (_type_): _description_

    Returns:
        _type_: _description_
    """
    
    # customize the boxplot
    p = 0
    for box in ax.patches:
        if box.__class__.__name__ == "PathPatch":
            
            # a list item for each layer group
            # remove alpha parameter
            color = box.get_facecolor()
            color = list(color)
            color.pop()
            color = tuple(color)
            
            # Each box has 6 associated Line2D objects
            # (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same colour
            # as above
            # group 1 (NV)
            for k in range(6 * p, 6 * (p + 1)):
                ax.lines[k].set_color(color)  # box
                ax.lines[k].set_mfc("w")      # whisker
                ax.lines[k].set_mec(color)    # fliers
            p += 1
            # set box edge and facecolor with 
            # separate alpha
            box.set_facecolor(color + (alpha,))
            box.set_edgecolor(color + (1,))
    return ax

    
def plot_fr_std_stats_by_layer(ax, df_nv, df_ns, df_ne, df_nb, layers, cl):
    
    N_BOOT = 100

    # initialize variables
    log_fr_vars_vivo = []
    log_fr_vars_sili_sp = []
    log_fr_vars_sili_ev = []
    layer_vivo = []
    layer_sili_sp = []
    layer_sili_ev = []
    
    # plot each layer
    for ix, layer in enumerate(layers):

        # vivo
        log_fr_vars_i = bootstrap_varlogfr(
            df_nv["firing_rate"][df_nv["layer"] == layer].astype(np.float32).tolist(),
            N_BOOT,
        )
        log_fr_vars_vivo += log_fr_vars_i
        layer_vivo += [layer] * len(log_fr_vars_i)

        # silico spontaneous
        log_fr_vars_sp_i = bootstrap_varlogfr(
            df_ns["firing_rate"][df_ns["layer"] == layer].astype(np.float32).tolist(),
            N_BOOT,
        )
        log_fr_vars_sili_sp += log_fr_vars_sp_i
        layer_sili_sp += [layer] * len(log_fr_vars_sp_i)

        # silico evoked
        log_fr_vars_ev_i = bootstrap_varlogfr(
            df_ne["firing_rate"][df_ne["layer"] == layer].astype(np.float32).tolist(),
            N_BOOT,
        )
        log_fr_vars_sili_ev += log_fr_vars_ev_i
        layer_sili_ev += [layer] * len(log_fr_vars_ev_i)


    # Format dataset to plot
    # Marques-Smith
    vivo_data = pd.DataFrame(
        data=np.array(log_fr_vars_vivo), columns=["Var(log(firing rate))"]
    )
    vivo_data["experiment"] = "M"
    vivo_data["layer"] = layer_vivo

    # spontaneous
    sili_data_sp = pd.DataFrame(
        data=np.array(log_fr_vars_sili_sp), columns=["Var(log(firing rate))"]
    )
    sili_data_sp["experiment"] = "NS"
    sili_data_sp["layer"] = layer_sili_sp

    # evoked
    sili_data_ev = pd.DataFrame(
        data=np.array(log_fr_vars_sili_ev), columns=["Var(log(firing rate))"]
    )
    sili_data_ev["experiment"] = "E"
    sili_data_ev["layer"] = layer_sili_ev

    # synthetic
    log_fr_vars_sili_nb = bootstrap_varlogfr(
        df_nb["firing_rate"].astype(np.float32).tolist(),
        N_BOOT,
    )
    sili_data_nb = pd.DataFrame(
        data=np.array(log_fr_vars_sili_nb), columns=["Var(log(firing rate))"]
    )
    sili_data_nb["experiment"] = "S"
    sili_data_nb["layer"] = "L5"

    # concat
    plot_data = pd.concat(
        [vivo_data, sili_data_sp, sili_data_ev, sili_data_nb], ignore_index=True
    )

    # drop sites outside layers
    mask = np.isin(plot_data["layer"], layers)
    plot_data = plot_data[mask]
    plot_data = plot_data.sort_values(by=["layer"])

    # we plot stds stats over bootstrapped log10(firing rate) which reflects
    # best what we see from the distribution plots (stats over raw data
    # is not visible).
    # plot
    ax = sns.boxplot(
        ax=ax,
        data=plot_data,
        x="layer",
        y="Var(log(firing rate))",
        hue="experiment",
        hue_order=["M", "NS", "E", "S"],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_NB"]],
        **BOXPLOT_PMS
    )

    # axes
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.set_ylabel("Var of log(firing rate)")
    ax.set_xlabel("Layer")

    # customize the boxplot
    p = 0
    for box in ax.patches:
        if box.__class__.__name__ == "PathPatch":
            # a list item for each layer group
            color = box.get_facecolor()           
            box.set_edgecolor(color)
            # Each box has 6 associated Line2D objects
            # (to make the whiskers, fliers, etc.)
            # Loop over them here, and use the same colour
            # as above
            # group 1 (NV)
            for k in range(6 * p, 6 * (p + 1)):
                ax.lines[k].set_color(color)  # box
                ax.lines[k].set_mfc(color)  # whisker
                ax.lines[k].set_mec(color)  # fliers
            p += 1
    ax.set_ylim(bottom=0)
    return ax, plot_data


def plot_fr_std_stats_by_layer_vert(ax, df_nv, df_ns, df_ne, df_nb, layers, cl):
    
    # 100 bootstrap
    N_BOOT = 100
    
    # plot parameters
    BOXPLOT_PMS = {
        "notch": True, # notches are 95% CI
        "gap": 0.2,
        "width": 0.7,
        "linewidth": 0.5,
        "flierprops": {
                "marker": "o",
                "markerfacecolor": "w",
                "markersize": 1.5,
                "markeredgewidth": 0.5
            },
        "boxprops": {"linewidth": 0.5}
    }
    alpha = 0.4

    # initialize variables
    log_fr_vars_vivo = []
    log_fr_vars_sili_sp = []
    log_fr_vars_sili_ev = []
    layer_vivo = []
    layer_sili_sp = []
    layer_sili_ev = []
    
    # plot each layer
    for ix, layer in enumerate(layers):

        # vivo
        log_fr_vars_i = bootstrap_varlogfr(
            df_nv["firing_rate"][df_nv["layer"] == layer].astype(np.float32).tolist(),
            N_BOOT,
        )
        log_fr_vars_vivo += log_fr_vars_i
        layer_vivo += [layer] * len(log_fr_vars_i)

        # silico spontaneous
        log_fr_vars_sp_i = bootstrap_varlogfr(
            df_ns["firing_rate"][df_ns["layer"] == layer].astype(np.float32).tolist(),
            N_BOOT,
        )
        log_fr_vars_sili_sp += log_fr_vars_sp_i
        layer_sili_sp += [layer] * len(log_fr_vars_sp_i)

        # silico evoked
        log_fr_vars_ev_i = bootstrap_varlogfr(
            df_ne["firing_rate"][df_ne["layer"] == layer].astype(np.float32).tolist(),
            N_BOOT,
        )
        log_fr_vars_sili_ev += log_fr_vars_ev_i
        layer_sili_ev += [layer] * len(log_fr_vars_ev_i)


    # Format dataset to plot
    # Marques-Smith
    vivo_data = pd.DataFrame(
        data=np.array(log_fr_vars_vivo), columns=["Var(log(firing rate))"]
    )
    vivo_data["experiment"] = "M"
    vivo_data["layer"] = layer_vivo

    # spontaneous
    sili_data_sp = pd.DataFrame(
        data=np.array(log_fr_vars_sili_sp), columns=["Var(log(firing rate))"]
    )
    sili_data_sp["experiment"] = "NS"
    sili_data_sp["layer"] = layer_sili_sp

    # evoked
    sili_data_ev = pd.DataFrame(
        data=np.array(log_fr_vars_sili_ev), columns=["Var(log(firing rate))"]
    )
    sili_data_ev["experiment"] = "E"
    sili_data_ev["layer"] = layer_sili_ev

    # synthetic
    log_fr_vars_sili_nb = bootstrap_varlogfr(
        df_nb["firing_rate"].astype(np.float32).tolist(),
        N_BOOT,
    )
    sili_data_nb = pd.DataFrame(
        data=np.array(log_fr_vars_sili_nb), columns=["Var(log(firing rate))"]
    )
    sili_data_nb["experiment"] = "S"
    sili_data_nb["layer"] = "L5"

    # concat
    plot_data = pd.concat(
        [vivo_data, sili_data_sp, sili_data_ev, sili_data_nb], ignore_index=True
    )

    # drop sites outside layers
    mask = np.isin(plot_data["layer"], layers)
    plot_data = plot_data[mask]
    plot_data = plot_data.sort_values(by=["layer"])

    # we plot stds stats over bootstrapped log10(firing rate) which reflects
    # best what we see from the distribution plots (stats over raw data
    # is not visible).
    # plot
    ax = sns.boxplot(
        ax=ax,
        data=plot_data,
        x="Var(log(firing rate))",
        y="layer",
        hue="experiment",
        hue_order=["M", "NS", "E", "S"],
        palette=[cl["COLOR_NV"], cl["COLOR_NS"], cl["COLOR_NE"], cl["COLOR_NB"]],
        **BOXPLOT_PMS,
        vert=False
    )

    # axes
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.set_xlabel("Var of log(firing rate)")
    ax.set_ylabel("Layer")

    # customize the boxplot
    ax = set_aes_boxes(ax, alpha)
    #ax.set_ylim(bottom=0)
    return ax, plot_data


def plot_single_unit_ratio(ax, df_vivo, df_silico_sp, df_silico_sp_2X, df_silico_ev, df_silico_nb, legend_cfg):
    """plot the proportions of single and multi-units
    sorted from the neuropixels probe recording
    """
    # setup figure
    shift = 0.3
    text_xpos_vivo = -0.3 + shift
    text_xpos_sili_sp = 0.7 + shift
    text_xpos_sili_sp_2X = 1.7 + shift
    text_xpos_sili_ev = 2.7 + shift
    text_xpos_sili_nb = 3.7 + shift

    # multi-unit and single unit
    # colors
    color = np.array(
        [
            [1, 1, 1],
            [0.2, 0.2, 0.2],
        ]
    )

    # single-unit count
    n_single_units_vivo = sum(df_vivo["kslabel"] == "good")
    n_single_units_sili_sp = sum(df_silico_sp["kslabel"] == "good")
    n_single_units_sili_sp_2X = sum(df_silico_sp_2X["kslabel"] == "good")
    n_single_units_sili_ev = sum(df_silico_ev["kslabel"] == "good")
    n_single_units_sili_nb = sum(df_silico_nb["kslabel"] == "good")
    
    # multi-unit count
    n_mu_vivo = df_vivo.shape[0] - n_single_units_vivo
    n_mu_sili_sp = df_silico_sp.shape[0] - n_single_units_sili_sp
    n_mu_sili_sp_2X = df_silico_sp_2X.shape[0] - n_single_units_sili_sp_2X
    n_mu_sili_ev = df_silico_ev.shape[0] - n_single_units_sili_ev
    n_mu_sili_nb = df_silico_nb.shape[0] - n_single_units_sili_nb

    # build dataset
    df = pd.DataFrame()
    df["M"] = np.array([n_single_units_vivo, n_mu_vivo]) / df_vivo.shape[0]
    df["NS"] = (
        np.array([n_single_units_sili_sp, n_mu_sili_sp]) / df_silico_sp.shape[0]
    )
    df["NS-2X"] = (
        np.array([n_single_units_sili_sp_2X, n_mu_sili_sp_2X]) / df_silico_sp_2X.shape[0]
    )    
    df["E"] = (
        np.array([n_single_units_sili_ev, n_mu_sili_ev]) / df_silico_ev.shape[0]
    )
    df["S"] = (
        np.array([n_single_units_sili_nb, n_mu_sili_nb]) / df_silico_nb.shape[0]
    )

    # bar plot
    df.T.plot.bar(
        ax=ax, stacked=True, color=color, edgecolor=(0.5, 0.5, 0.5), rot=0, width=0.8, linewidth=0.5
    )
    
    # add unit counts
    ax.annotate(
        f"""{n_mu_vivo}""",
        (text_xpos_vivo, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_vivo}""",
        (text_xpos_vivo, 0.13),
        ha="center",
        color="k",
        rotation=0,
    )    
    # Spontaneous NPX model
    ax.annotate(
        f"""{n_mu_sili_sp}""",
        (text_xpos_sili_sp, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_sili_sp}""",
        (text_xpos_sili_sp, 0.13),
        ha="center",
        color="k",
        rotation=0,
    )
    # 2X
    ax.annotate(
        f"""{n_mu_sili_sp_2X}""",
        (text_xpos_sili_sp_2X, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_sili_sp_2X}""",
        (text_xpos_sili_sp_2X, 0.13),
        ha="center",
        color="k",
        rotation=0,
    )    
    # evoked 
    ax.annotate(
        f"""{n_mu_sili_ev}""",
        (text_xpos_sili_ev, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_sili_ev}""",
        (text_xpos_sili_ev, 0.02),
        ha="center",
        color="k",
        rotation=0,
    )
    # synthetic
    ax.annotate(
        f"""{n_mu_sili_nb}""",
        (text_xpos_sili_nb, 0.8),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_single_units_sili_nb}""",
        (text_xpos_sili_nb, 0.1),
        ha="center",
        color="k",
        rotation=0,
    )    
    ax.legend(
        ["single-unit", "multi-unit"],
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        **legend_cfg,
    )
    ax.set_ylabel("Proportion (ratio)")
    ax.set_xlabel("Experiment")
    return ax


def plot_single_unit_ratio_by_drift_corr(ax, drift_corr_v100, drift_corr_rtx5090, no_drift_corr_rtx5090, legend_cfg, number_pos: dict):
    """plot the proportions of single and multi-units
    sorted from the neuropixels probe recording for three 
    conditions
    """
    # colors
    color = np.array(
        [
            [1, 1, 1], # single-units
            [0.2, 0.2, 0.2], # multi-units
        ]
    )

    # single-unit count
    n_su_1 = sum(drift_corr_v100["kslabel"] == "good")
    n_su_2 = sum(drift_corr_rtx5090["kslabel"] == "good")
    n_su_3 = sum(no_drift_corr_rtx5090["kslabel"] == "good")
    
    # multi-unit count
    n_mu_1 = drift_corr_v100.shape[0] - n_su_1
    n_mu_2 = drift_corr_rtx5090.shape[0] - n_su_2
    n_mu_3 = no_drift_corr_rtx5090.shape[0] - n_su_3

    # build dataset
    df = pd.DataFrame()
    df["Corrected \n(V100)"] = np.array([n_su_1, n_mu_1]) / drift_corr_v100.shape[0]
    df["Corrected \n(rtx5090)"] = (
        np.array([n_su_2, n_mu_2]) / drift_corr_rtx5090.shape[0]
    )
    df["Uncorrected \n(rtx5090)"] = (
        np.array([n_su_3, n_mu_3]) / no_drift_corr_rtx5090.shape[0]
    )    

    # bar plot
    df.T.plot.bar(
        ax=ax, stacked=True, color=color, edgecolor=(0.5, 0.5, 0.5), rot=0, width=0.8, linewidth=0.5
    )
    
    # add unit counts
    ax.annotate(
        f"""{n_mu_1}""",
        (number_pos['exp1_x'], number_pos['exp1_y_mu']),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_su_1}""",
        (number_pos['exp1_x'], number_pos['exp1_y_su']),
        ha="center",
        color="k",
        rotation=0,
    )    
    # Spontaneous NPX model
    ax.annotate(
        f"""{n_mu_2}""",
        (number_pos['exp2_x'], number_pos['exp2_y_mu']),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_su_2}""",
        (number_pos['exp2_x'], number_pos['exp2_y_su']),
        ha="center",
        color="k",
        rotation=0,
    )
    # 2X
    ax.annotate(
        f"""{n_mu_3}""",
        (number_pos['exp3_x'], number_pos['exp3_y_mu']),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_su_3}""",
        (number_pos['exp3_x'], number_pos['exp3_y_su']),
        ha="center",
        color="k",
        rotation=0,
    )    
    ax.legend(
        ["single-unit", "multi-unit"],
        loc="upper left",
        bbox_to_anchor=(0, 1.25),
        **legend_cfg,
    )
    ax.set_ylabel("Proportion (ratio)")
    ax.set_xlabel("Drift correction")
    return ax


def get_stats(df_fr, value_col, layers, exps_order):
    """Calculate Mann-Whitney U test with method set 
    to "auto".
    
    note: "auto" is much faster than "exact" for large 
    sample size
    """

    myhash = collections.defaultdict(dict)

    for li, ly in enumerate(layers):
        for ei, targ_exp in enumerate(exps_order[1:]):

            # get experiments data
            M = df_fr[(df_fr["layer"] == ly) & (df_fr["experiment"] == "M")]
            target = df_fr[(df_fr["layer"] == ly) & (df_fr["experiment"] == targ_exp)]

            # Experiments vs Marques-Smith
            if (len(M[value_col]) > 0) & (len(target[value_col]) > 0):
                
                # calculate Mann-Whitney U test
                z, p = mannwhitneyu(
                    M[value_col].values,
                    target[value_col].values,
                    method="auto",
                )
                print(
                    f"""{ly} - M vs. {targ_exp}, z={z}, p={p}, N_m={len(M[value_col].values)}, N_target={len(target[value_col].values)}"""
                )
                
                # record stats in hash
                myhash[ly][targ_exp] = {}
                myhash[ly][targ_exp]["p"] = p
                myhash[ly][targ_exp]["z"] = z
                myhash[ly][targ_exp]["symbol"] = (
                    ""
                    if np.isnan(p)
                    else
                    "ns"
                    if p >= 0.05
                    else (
                        "*"
                        if p < 0.05 and p >= 0.01
                        else "**" if p < 0.01 and p >= 0.001 else "***"
                    )
                )
            # case no data, record empty symbol
            else:
                print(
                    f"""{ly} - M vs. {targ_exp}, z="nan", p="nan", N_m={len(M[value_col].values)}, N_target={len(target[value_col].values)}"""
                )
                myhash[ly][targ_exp] = {}
                myhash[ly][targ_exp]["symbol"] = ""
    return myhash


def annotate_stats(ax, df, value_col, layers, exps_order, xadj=1):
    """Annotate plot with statistical significance symbols 
    for the Mann-Whitney U test

    Args:
        ax (_type_): _description_
        df (_type_): _description_
        value_col (_type_): _description_
        layers (_type_): _description_
        exps_order (_type_): _description_
        xadj (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """
    # plot parameters
    FTZ = 5

    # get Mann-Whitney stats
    stats_hash = get_stats(df, value_col, layers, exps_order)

    # x-coord boxes
    GAP = 0.17

    NS_xc = np.array([0, 1, 2, 3, 4])
    M_xc = NS_xc - GAP
    E_xc = NS_xc + GAP
    S_xc = E_xc + GAP
    
    # max data value
    maxv = df[value_col].max()

    # annotate stats
    for li, ly in enumerate(layers):
        
        # NS *************
        # plot nothing
        if stats_hash[ly]["NS"]["symbol"] == "":
            continue
        else:
            # line
            ax.vlines(
                x=maxv + xadj * 0.069 * maxv,
                ymin=M_xc[li] - 0.1,
                ymax=NS_xc[li] - 0.05,
                color="k",
                linewidth=0.5,
            )
            # symbol
            adj = 0
            if stats_hash[ly]["NS"]["symbol"] == "ns":
                adj = xadj*0.069 * maxv
                
            ax.annotate(
                text=stats_hash[ly]["NS"]["symbol"],
                xy=(maxv - xadj*0.055 * maxv + adj, M_xc[li] + 0.5 * GAP),
                color="k",
                fontsize=FTZ,
                rotation=270,
            )

        # E *************
        if stats_hash[ly]["E"]["symbol"] == "":
            continue
        else:
            # line
            ax.vlines(
                x=maxv + xadj * 0.278 * maxv,
                ymin=M_xc[li] - 0.1,
                ymax=E_xc[li] - 0.05,
                color="k",
                linewidth=0.5,
            )
            # symbol
            adj = 0
            if stats_hash[ly]["E"]["symbol"] == "ns":
                adj = xadj*0.069 * maxv
            ax.annotate(
                text=stats_hash[ly]["E"]["symbol"],
                xy=(maxv + xadj * 0.16 * maxv + adj, M_xc[li] + GAP),
                color="k",
                fontsize=FTZ,
                rotation=270,
            )

        # S *************
        if stats_hash[ly]["S"]["symbol"] == "":
            continue
        else:
            ax.vlines(
                x=maxv + xadj * 0.488 * maxv,
                ymin=M_xc[li] - 0.1,
                ymax=S_xc[li] - 0.05,
                color="k",
                linewidth=0.5,
            )
            # annotate with significance symbol
            adj = 0
            if stats_hash[ly]["S"]["symbol"] == "ns":
                adj = xadj * 0.069 * maxv
            ax.annotate(
                text=stats_hash[ly]["S"]["symbol"],
                xy=(maxv + xadj * 0.4 * maxv + adj, M_xc[li] + 1.5 * GAP),
                color="k",
                fontsize=FTZ,
                rotation=270,
            )
    return ax, stats_hash