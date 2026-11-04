"""Statistics of Fig. 5a, written next to its source data

author: laquitainesteeve@gmail.com

Reads figures/8_source_data/fig5a/fig5a_npx.xlsx and fig5a_dense.xlsx (number of single-units and multi-units of each experiment) and writes
fig5a_statistics.xlsx (merged into Source_Data.xlsx by merge_source_data.py). The tests are those printed on the figure, computed as in the paper's
notebook (notebooks/2_results_editorial_requests/fig5a.ipynb): Pearson's chi-square goodness-of-fit test (scipy.stats.chisquare, expected frequencies =
mean) applied to the percentage of single-units of the compared experiments (the p values depend on this scaling).

Usage:

    PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/add_statistics_fig5.py
"""

import numpy as np
import pandas as pd
from scipy import stats

SOURCE = "figures/8_source_data/fig5a"
TESTS = {"M, NS, NS-2X": ["M", "NS", "NS-2X"], "NS, E": ["NS", "E"], "M, S": ["M", "S"], "H, DS": ["H", "DS"], "M, H": ["M", "H"], "NS, DS": ["NS", "DS"]}


def main():
    d = pd.concat([pd.read_excel(f"{SOURCE}/fig5a_npx.xlsx"), pd.read_excel(f"{SOURCE}/fig5a_dense.xlsx")])
    single = d[d["unit_type"] == "single-unit"].set_index("experiment")["n_units"]
    multi = d[d["unit_type"] == "multi-unit"].set_index("experiment")["n_units"]
    rows = []
    for name, ks in TESTS.items():
        pct = np.array([single[k] / (single[k] + multi[k]) * 100 for k in ks])
        g = stats.chisquare(f_obs=pct)
        rows.append(dict(compared_experiments=name, n_units=int(sum(single[k] + multi[k] for k in ks)), df=len(ks) - 1,
                         chi_square_goodness_of_fit_on_percentages=g.statistic, p_value_goodness_of_fit_on_percentages=g.pvalue))
    out = pd.DataFrame(rows)
    out.to_excel(f"{SOURCE}/fig5a_statistics.xlsx", index=False)
    print(out.round(4).to_string())


if __name__ == "__main__":
    main()
