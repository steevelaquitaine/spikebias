import numpy as np
import pandas as pd

        
def plot_single_unit_ratio(
    ax, df_hv1, df_hv2, df_hv3, df_hs1, df_hs2, df_hs3
):
    """plot the proportions of single and multi-units
    sorted from the dense probe recording
    """

    # setup figure
    shift = 0.3
    text_xpos_vivo = -0.3 + shift
    text_xpos_sili_sp = 0.7 + shift

    # multi-unit and single unit
    # colors
    color = np.array(
        [
            [1, 1, 1],
            [0.2, 0.2, 0.2],
        ]
    )

    # number of single-units
    # vivo
    n_su_hv1 = sum(df_hv1["kslabel"] == "good")
    n_su_hv2 = sum(df_hv2["kslabel"] == "good")
    n_su_hv3 = sum(df_hv3["kslabel"] == "good")    
    # biophy
    n_su_hs1 = sum(df_hs1["kslabel"] == "good")
    n_su_hs2 = sum(df_hs2["kslabel"] == "good")
    n_su_hs3 = sum(df_hs3["kslabel"] == "good")

    # number of multi-units
    # vivo
    n_mu_hv1 = df_hv1.shape[0] - n_su_hv1
    n_mu_hv2 = df_hv2.shape[0] - n_su_hv2
    n_mu_hv3 = df_hv3.shape[0] - n_su_hv3
    # biophy
    n_mu_hs1 = df_hs1.shape[0] - n_su_hs1
    n_mu_hs2 = df_hs2.shape[0] - n_su_hs2
    n_mu_hs3 = df_hs3.shape[0] - n_su_hs3
    
    # totals
    # single-units
    n_su_hv = n_su_hv1 + n_su_hv2 + n_su_hv3
    n_su_hs = n_su_hs1 + n_su_hs2 + n_su_hs3
    # multi-units
    n_mu_hv = n_mu_hv1 + n_mu_hv2 + n_mu_hv3
    n_mu_hs = n_mu_hs1 + n_mu_hs2 + n_mu_hs3
    # totals
    n_hv = df_hv1.shape[0] + df_hv2.shape[0] + df_hv3.shape[0]
    n_hs = df_hs1.shape[0] + df_hs2.shape[0] + df_hs3.shape[0]

    # build dataset
    df = pd.DataFrame()
    df["H"] = np.array([n_su_hv, n_mu_hv]) / n_hv
    df["DS"] = np.array([n_su_hs, n_mu_hs]) / n_hs

    # bar plot
    df.T.plot.bar(
        ax=ax,
        stacked=True,
        color=color,
        edgecolor=(0.5, 0.5, 0.5),
        rot=0,
        width=0.8,
        linewidth=0.5,
    )

    # label unit counts
    # vivo
    ax.annotate(
        f"""{n_mu_hv}""",
        (text_xpos_vivo, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_su_hv}""",
        (text_xpos_vivo, 0.13),
        ha="center",
        color="k",
        rotation=0,
    )
    # biophy
    ax.annotate(
        f"""{n_mu_hs}""",
        (text_xpos_sili_sp, 0.6),
        ha="center",
        color="w",
        rotation=0,
    )
    ax.annotate(
        f"""{n_su_hs}""",
        (text_xpos_sili_sp, 0.13),
        ha="center",
        color="k",
        rotation=0,
    )
    ax.set_ylabel("Proportion (ratio)")
    ax.set_xlabel("Experiment")
    return ax


def plot_single_unit_ratio_with_df(
    ax, df_hv1, df_hv2, df_hv3, df_hs1, df_hs2, df_hs3
):
    """same as plot_single_unit_ratio() but also returns the tidy
    source dataset (one row per experiment x unit type) used to draw
    the figure, for export as a source dataset
    """
    ax = plot_single_unit_ratio(ax, df_hv1, df_hv2, df_hv3, df_hs1, df_hs2, df_hs3)

    n_su_hv = sum(sum(d["kslabel"] == "good") for d in (df_hv1, df_hv2, df_hv3))
    n_su_hs = sum(sum(d["kslabel"] == "good") for d in (df_hs1, df_hs2, df_hs3))
    n_hv = df_hv1.shape[0] + df_hv2.shape[0] + df_hv3.shape[0]
    n_hs = df_hs1.shape[0] + df_hs2.shape[0] + df_hs3.shape[0]
    n_mu_hv = n_hv - n_su_hv
    n_mu_hs = n_hs - n_su_hs

    n_single = {"H": n_su_hv, "DS": n_su_hs}
    n_multi = {"H": n_mu_hv, "DS": n_mu_hs}
    n_total = {"H": n_hv, "DS": n_hs}

    df_source = pd.DataFrame(
        [
            {
                "experiment": exp,
                "unit_type": unit_type,
                "n_units": n_map[exp],
                "ratio": n_map[exp] / n_total[exp],
            }
            for exp in ("H", "DS")
            for unit_type, n_map in [("single-unit", n_single), ("multi-unit", n_multi)]
        ]
    )
    return ax, df_source