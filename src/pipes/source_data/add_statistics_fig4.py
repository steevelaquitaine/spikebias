"""Statistics of Fig. 4b-c, written next to their source data

author: laquitainesteeve@gmail.com

Reads figures/8_source_data/fig4b/fig4b.xlsx (log10 firing rates of the single units) and fig4c/fig4c.xlsx (variances of the log10 firing rates, 100
bootstraps per box) and writes fig4b_statistics.xlsx and fig4c_statistics.xlsx (merged into Source_Data.xlsx by merge_source_data.py) with:
* sheet "per_layer": two-sided Mann-Whitney U test of each simulation (NS: spontaneous, E: evoked, S: synthetic) against the Marques-Smith data (M) in each
  layer (the tests of the stars of the figure; no correction for multiple comparisons): n, medians, difference of the medians (simulation - M) with its
  95% bootstrap confidence interval (10,000 resamples, seed 0; spikes/s in 4b, variance in 4c), U statistic of M, exact P, symbol
* sheet "layers_5_6" (4b only): the test of the Results text: median firing rate in layers 5 and 6 (spikes/s) of M, of the simulations and of NS and E pooled,
  difference of the medians with its 95% bootstrap confidence interval (10,000 resamples, seed 0) and two-sided Mann-Whitney U test

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics_fig4.py
"""

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from src.nodes.validation.firing_rate_plots import LAYERS, PANELS, symbol

SOURCE = "figures/8_source_data"
SEED = 0
N_BOOT = 10_000


def median_difference(m, x, to_rate, rng):
    """Difference of the medians (x - m) and its 95% bootstrap confidence interval, in the unit given by to_rate"""
    m, x = to_rate(np.asarray(m)), to_rate(np.asarray(x))
    boot = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(m, len(m))) for _ in range(N_BOOT)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return np.median(x) - np.median(m), lo, hi


def per_layer(df, value, to_rate):
    rng = np.random.default_rng(SEED)
    rows = []
    for layer in LAYERS:
        m = df[(df["layer"] == layer) & (df["experiment"] == "M")][value].dropna().values
        for exp in ["NS", "E", "S"]:
            x = df[(df["layer"] == layer) & (df["experiment"] == exp)][value].dropna().values
            if len(m) == 0 or len(x) == 0:
                continue
            res = mannwhitneyu(m, x, alternative="two-sided", method="auto")
            diff, lo, hi = median_difference(m, x, to_rate, rng)
            rows.append(dict(layer=layer, comparison=f"{exp} vs M", test="Mann-Whitney U (two-sided)", n_M=len(m), n_simulation=len(x),
                             median_M=to_rate(np.median(m)), median_simulation=to_rate(np.median(x)), difference_of_medians=diff, ci95_low=lo, ci95_high=hi,
                             U_M=res.statistic, p_value=res.pvalue,
                             significance=symbol(res.pvalue)))
    return pd.DataFrame(rows)


def layers_5_6(df, value):
    rng = np.random.default_rng(SEED)
    rate = lambda e: 10 ** df[df["experiment"].isin(e) & df["layer"].isin(["L5", "L6"])][value].dropna().values   # spikes/s
    m = rate(["M"])
    rows = []
    for name, exps in (("NS vs M", ["NS"]), ("E vs M", ["E"]), ("NS and E pooled vs M", ["NS", "E"]), ("S vs M", ["S"])):
        x = rate(exps)
        diff = np.median(x) - np.median(m)
        boot = [np.median(rng.choice(x, len(x))) - np.median(rng.choice(m, len(m))) for _ in range(N_BOOT)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        res = mannwhitneyu(m, x, alternative="two-sided", method="auto")
        rows.append(dict(comparison=name, layers="L5 and L6", n_M=len(m), n_simulation=len(x), median_M_spikes_per_s=np.median(m),
                         median_simulation_spikes_per_s=np.median(x), difference_spikes_per_s=diff, ci95_low=lo, ci95_high=hi, test="Mann-Whitney U (two-sided)",
                         U_M=res.statistic, p_value=res.pvalue))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    for panel, to_rate in (("b", lambda v: 10 ** v), ("c", lambda v: v)):
        value = PANELS[panel]["value"]
        df = pd.read_excel(f"{SOURCE}/fig4{panel}/fig4{panel}.xlsx")
        pl = per_layer(df, value, to_rate)
        with pd.ExcelWriter(f"{SOURCE}/fig4{panel}/fig4{panel}_statistics.xlsx") as w:
            pl.to_excel(w, sheet_name="per_layer", index=False)
            if panel == "b":
                layers_5_6(df, value).to_excel(w, sheet_name="layers_5_6", index=False)
        print(f"fig4{panel}: {len(pl)} tests, smallest P {pl.p_value.min():.1e}, largest P {pl.p_value.max():.1e}")
    print(pd.read_excel(f"{SOURCE}/fig4b/fig4b_statistics.xlsx", sheet_name="layers_5_6").round(3).to_string())
