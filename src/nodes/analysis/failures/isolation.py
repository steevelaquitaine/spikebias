
from src.nodes.validation.npx_probe import firing_rate as npx_fr
from src.nodes.validation.dense_probe import firing_rate as dense_fr


def plot_unit_isolation(ax, df_nv, df_ns, df_ns_2X, df_ne, df_nb, legend_cfg: dict):
    """stacked bar plot of the ratio of sorted single-units and multi-units

    Args:
        ax (_type_): _description_
        df_nv (_type_): _description_
        df_ns (_type_): _description_
        df_ne (_type_): _description_
        df_nb (_type_): _description_
        legend_cfg (dict): _description_

    Returns:
        _type_: _description_
    """
    # plot
    ax = npx_fr.plot_single_unit_ratio(ax, df_nv, df_ns, df_ns_2X, df_ne, df_nb, legend_cfg)

    # esthetics
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    return ax


def plot_unit_isolation_dense_probe(ax, df_hv1, df_hv2, df_hv3, df_hs1, df_hs2, df_hs3):
    
    # plot
    ax = dense_fr.plot_single_unit_ratio(
        ax, df_hv1, df_hv2, df_hv3, df_hs1, df_hs2, df_hs3
    )
    # esthetics
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend("", frameon=False)

    # disconnect axes (R style)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    return ax