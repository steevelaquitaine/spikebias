"""Single-unit and multi-unit counts of Supplementary Fig. 9 (effect of the drift correction on the yield of Kilosort 4, 3 and 2.5)

author: laquitainesteeve@gmail.com

The figure annotates every bar with its number of sorted units. These numbers are the text of the paper's pdf (vector text, nothing is
digitized from pixels) and are read with `pdftotext -bbox`. The sortings of the two RTX 5090 columns are not stored anywhere, so these counts
are the only record of them; the V100 column is also recomputed from the sortings in dataset/01_intermediate/sorting/ for a check.
"""

import re
import subprocess

import numpy as np
import pandas as pd
import spikeinterface as si

SORTERS = ["ks4", "ks3", "ks2.5"]               # rows
DATASETS = ["M", "NS", "S"]                     # columns
CONDITIONS = ["+ (V100)", "+ (rtx5090)", "- (rtx5090)"]
LAYERS = ["L1", "L2/3", "L4", "L5", "L6"]

# bars: x position (pt) of the bar centers in the paper's page; the panels are 147 pt apart, the bars 31.7 pt
X0, X_PANEL, X_BAR = 110.4, 147.0, 31.7
ROW_SPLITS = (335, 430)                          # y (pt) separating the three rows of panels

# V100 sortings (10 minutes) and the layers to keep
V100_PATHS = {
    "M": "dataset/01_intermediate/sorting/marques/Sorting{}_10m",
    "NS": "dataset/01_intermediate/sorting/npx_spont/Sorting{}_10m",
    "S": "dataset/01_intermediate/sorting/buccino/Sorting{}_10m",
}
SORTER_FOLDER = {"ks4": "KS4", "ks3": "KS3", "ks2.5": "KS2_5"}


def extract_counts_from_paper_pdf(pdf_path: str, page: int = 47) -> pd.DataFrame:
    """Counts of single-units and multi-units annotated on each bar of the paper's Supplementary Fig. 9 (the multi-unit count is in the
    black part of the bar, above the single-unit count)"""
    xml = subprocess.run(["pdftotext", "-f", str(page), "-l", str(page), "-bbox", pdf_path, "-"], capture_output=True, text=True, check=True).stdout
    words = [(float(a), float(b), float(c), float(d), w) for a, b, c, d, w in
             re.findall(r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" yMax="([\d.]+)">(\d+)</word>', xml)]
    groups = {}
    for xmin, ymin, xmax, ymax, w in words:
        xc = (xmin + xmax) / 2
        panel_bar = [(p, b) for p in range(3) for b in range(3) if abs(xc - (X0 + p * X_PANEL + b * X_BAR)) < 8]
        if not panel_bar or ymin < 250 or ymin > 500:
            continue
        row = int(ymin > ROW_SPLITS[0]) + int(ymin > ROW_SPLITS[1])
        groups.setdefault((row, *panel_bar[0]), []).append((ymin, int(w)))
    rows = []
    for (r, p, b), vals in sorted(groups.items()):
        assert len(vals) == 2, f"expected 2 counts for the bar {(r, p, b)}, found {vals}"
        (_, mu), (_, su) = sorted(vals)
        rows.append(dict(sorter=SORTERS[r], dataset=DATASETS[p], condition=CONDITIONS[b], single_units=su, multi_units=mu))
    df = pd.DataFrame(rows)
    assert len(df) == 27, "3 sorters x 3 datasets x 3 conditions"
    df["total_units"] = df["single_units"] + df["multi_units"]
    df["proportion_single_units"] = df["single_units"] / df["total_units"]
    df["proportion_multi_units"] = df["multi_units"] / df["total_units"]
    return df


def get_v100_counts() -> pd.DataFrame:
    """Single-units and multi-units of the V100 sortings (drift correction on) stored in dataset/01_intermediate/sorting/ (units in cortex)"""
    rows = []
    for sorter in SORTERS:
        for ds, path in V100_PATHS.items():
            S = si.load_extractor(path.format(SORTER_FOLDER[sorter]))
            label = np.array(S.get_property("KSLabel"))
            if ds != "S":                                     # all the units of Buccino's model are in layer 5
                layer = pd.Series(S.get_property("layer")).replace({"L2": "L2/3", "L3": "L2/3", "L2_3": "L2/3"}).values
                label = label[np.isin(layer, LAYERS)]
            rows.append(dict(sorter=sorter, dataset=ds, single_units_local=int((label == "good").sum()), multi_units_local=int((label != "good").sum())))
    return pd.DataFrame(rows)


# positions (x: bar index, y: proportion) of the unit counts in each panel (rows: ks4, ks3, ks2.5; columns: M, NS, S), as in the paper
NUMBER_Y = {  # (single-units, multi-units)
    ("ks4", "M"): (0.01, 0.6), ("ks4", "NS"): (0.01, 0.6), ("ks4", "S"): (0.01, 0.7),
    ("ks3", "M"): (0.1, 0.6), ("ks3", "NS"): (0.1, 0.6), ("ks3", "S"): (0.01, 0.7),
    ("ks2.5", "M"): (0.01, 0.6), ("ks2.5", "NS"): (0.1, 0.6), ("ks2.5", "S"): (0.01, 0.7),
}
# in the paper, the y label of each panel is the sorter in the first two columns (except ks2.5 / NS) and "Proportion (ratio)" otherwise
YLABEL_IS_SORTER = {("ks4", "M"), ("ks4", "NS"), ("ks3", "M"), ("ks3", "NS"), ("ks2.5", "M")}


def plot_source_data(df: pd.DataFrame, figsize=(6, 4)):
    """Stacked bars of the proportions of single-units (white) and multi-units (black) for each sorter (rows), dataset (columns) and
    condition (bars); each bar is annotated with the number of single-units and multi-units"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    for r, sorter in enumerate(SORTERS):
        for c, ds in enumerate(DATASETS):
            ax = axes[r, c]
            d = df[(df["sorter"] == sorter) & (df["dataset"] == ds)].set_index("condition").loc[CONDITIONS]
            x = np.arange(3)
            ax.bar(x, d["proportion_single_units"], width=0.8, color="w", edgecolor=(0.5, 0.5, 0.5), linewidth=0.5)
            ax.bar(x, d["proportion_multi_units"], width=0.8, bottom=d["proportion_single_units"], color=(0.2, 0.2, 0.2), edgecolor=(0.5, 0.5, 0.5), linewidth=0.5)
            y_su, y_mu = NUMBER_Y[(sorter, ds)]
            for i, cond in enumerate(CONDITIONS):
                ax.annotate(f"{int(d.loc[cond, 'multi_units'])}", (i, y_mu), ha="center", color="w")
                ax.annotate(f"{int(d.loc[cond, 'single_units'])}", (i, y_su), ha="center", color="k")
            ax.set_xticks(x)
            ax.set_xticklabels([c_.replace(" (", " \n(") for c_ in CONDITIONS], rotation=0)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1])
            ax.set_xlabel("Drift correction")
            ax.set_ylabel(sorter if (sorter, ds) in YLABEL_IS_SORTER else "Proportion (ratio)")
            if r == 0:
                ax.set_title(ds)
            ax.spines[["top", "right"]].set_visible(False)
            ax.spines["bottom"].set_position(("axes", -0.05))
            ax.yaxis.set_ticks_position("left")
            ax.spines["left"].set_position(("axes", -0.05))
    fig.tight_layout()
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="w", edgecolor=(0.5, 0.5, 0.5), linewidth=0.5),
               plt.Rectangle((0, 0), 1, 1, facecolor=(0.2, 0.2, 0.2), edgecolor=(0.5, 0.5, 0.5), linewidth=0.5)]
    axes[2, 2].legend(handles, ["single-unit", "multi-unit"], frameon=False, bbox_to_anchor=(1, 1), loc="upper left")
    return fig
