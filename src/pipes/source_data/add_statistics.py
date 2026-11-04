"""Statistics of the box-plot panels of Fig. 2 (m-p, r-u), written next to their source data

author: laquitainesteeve@gmail.com

For each panel, reads figures/8_source_data/fig2<panel>/fig2<panel>.xlsx and writes fig2<panel>_statistics.xlsx (merged into Source_Data.xlsx by
merge_source_data.py) with two sheets:
* "across_experiments": in each layer, Kruskal-Wallis test across experiments (statistic = chi-square, df, exact P; the synthetic model of Buccino,
  recorded in layer 5 only, is not in this test, as in the paper) and Dunn's post hoc tests between pairs of experiments (two-sided, Holm-Sidak
  adjusted exact P; all the experiments of the layer, i.e. including the synthetic model in layer 5)
* "across_layers": in each experiment, Kruskal-Wallis test across the layers (df = 4) and Dunn's post hoc tests between pairs of layers (Holm-Sidak)

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics.py
"""

import itertools
import os

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import kruskal

SOURCE = "figures/8_source_data"
LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]
# experiments of each panel, in the order of the figure (the synthetic model is last: layer 5 only)
PANELS = {
    "m": ["npx (Marques-Smith)", "Biophy. spont.", "Biophy. evoked", "Synth. (Buccino)"],
    "n": ["npx (Marques-Smith)", "Biophy. spont.", "Biophy. evoked", "Synth. (Buccino)"],
    "o": ["Dense (Horvath)", "Dense (Biophy. spontaneous)"],
    "p": ["Dense probe (Horvath)", "Dense probe (biophy. spont.)"],
    "r": ["M", "NS", "E", "S"],
    "s": ["MS", "NS", "NE", "NB"],
    "t": ["H", "DS"],
    "u": ["H", "DS"],
}


def columns(df):
    value = df.columns[0]
    layer = next(c for c in df.columns if c.lower() == "layer")
    experiment = next(c for c in df.columns if c.lower() == "experiment")
    return value, layer, experiment


def dunn(groups):
    """Dunn's post hoc test with Holm-Sidak correction: adjusted P of each pair"""
    p = sp.posthoc_dunn([g for g in groups.values()], p_adjust="holm-sidak")
    names = list(groups)
    return {(a, b): p.iloc[i, j] for (i, a), (j, b) in itertools.combinations(enumerate(names), 2)}


def statistics(panel):
    df = pd.read_excel(f"{SOURCE}/fig2{panel}/fig2{panel}.xlsx")
    value, layer_col, exp_col = columns(df)
    present = [e for e in PANELS[panel] if e in set(df[exp_col])]
    kw_exps = [e for e in present if df[df[exp_col] == e][layer_col].nunique() > 1]       # without the synthetic model (layer 5 only)

    def values(exp, layer):
        return df[(df[exp_col] == exp) & (df[layer_col] == layer)][value].dropna().values

    rows_exp = []
    for layer in LAYERS:
        h, p = kruskal(*[values(e, layer) for e in kw_exps])
        rows_exp.append(dict(layer=layer, test="Kruskal-Wallis", group_1=" / ".join(kw_exps), group_2="", n_1="/".join(str(len(values(e, layer))) for e in kw_exps),
                             n_2="", chi_square=h, df=len(kw_exps) - 1, p_value=p))
        groups = {e: values(e, layer) for e in present if len(values(e, layer)) > 0}
        for (a, b), padj in dunn(groups).items():
            rows_exp.append(dict(layer=layer, test="Dunn (two-sided, Holm-Sidak)", group_1=a, group_2=b, n_1=len(groups[a]), n_2=len(groups[b]),
                                 chi_square=np.nan, df=np.nan, p_value=padj))
    rows_lay = []
    for exp in [e for e in present if df[df[exp_col] == e][layer_col].nunique() > 1]:
        groups = {layer: values(exp, layer) for layer in LAYERS if len(values(exp, layer)) > 0}
        h, p = kruskal(*groups.values())
        rows_lay.append(dict(experiment=exp, test="Kruskal-Wallis", layer_1=" / ".join(groups), layer_2="", n_1="/".join(str(len(g)) for g in groups.values()),
                             n_2="", chi_square=h, df=len(groups) - 1, p_value=p))
        for (a, b), padj in dunn(groups).items():
            rows_lay.append(dict(experiment=exp, test="Dunn (two-sided, Holm-Sidak)", layer_1=a, layer_2=b, n_1=len(groups[a]), n_2=len(groups[b]),
                                 chi_square=np.nan, df=np.nan, p_value=padj))
    return pd.DataFrame(rows_exp), pd.DataFrame(rows_lay)


if __name__ == "__main__":
    summary = []
    for panel in PANELS:
        across_exp, across_lay = statistics(panel)
        with pd.ExcelWriter(f"{SOURCE}/fig2{panel}/fig2{panel}_statistics.xlsx") as writer:
            across_exp.to_excel(writer, sheet_name="across_experiments", index=False)
            across_lay.to_excel(writer, sheet_name="across_layers", index=False)
        kl = across_lay[across_lay["test"] == "Kruskal-Wallis"]
        ke = across_exp[across_exp["test"] == "Kruskal-Wallis"]
        summary.append((panel, ke.chi_square.min(), ke.chi_square.max(), ke.p_value.max(), kl.chi_square.min(), kl.chi_square.max(), kl.p_value.max(), kl.p_value.min()))
        print(f"fig2{panel}: across experiments chi2 {summary[-1][1]:.1f}-{summary[-1][2]:.1f} (largest P {summary[-1][3]:.2e}); across layers chi2 {summary[-1][4]:.1f}-{summary[-1][5]:.1f} "
              f"(largest P {summary[-1][6]:.2e}, smallest {summary[-1][7]:.2e})")
