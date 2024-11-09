import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def plot(ax, metric_data):
    
    # make arrays
    precisions = []
    recalls = []
    for m_i in metric_data:
        precisions.append(m_i["precision"])
        recalls.append(m_i["recall"])
        
    # plot performance
    df = pd.DataFrame(data=[precisions, recalls], index=["precision", "recall"]).T
    sns.stripplot(ax=ax, data=df, jitter=0.04, color="k", size=3)

    # stats precision
    ax.errorbar(
        x=0,
        y=np.nanmedian(precisions),
        yerr=1.96 * np.std(precisions) / np.sqrt(len(precisions)),  # 95% ci
        marker="o",
        color="orange",
        markeredgecolor="w",
        markersize=5,
        zorder=np.inf,
    )

    # stats recall
    ax.errorbar(
        x=1,
        y=np.nanmedian(recalls),
        yerr=1.96 * np.std(recalls) / np.sqrt(len(recalls)),  # 95% ci
        marker="o",
        color="orange",
        markeredgecolor="w",
        markersize=5,
        zorder=np.inf,
    )

    # disconnect axes (R style)
    ax.spines[["right", "top"]].set_visible(False)
    ax.spines["bottom"].set_position(("axes", -0.05))
    ax.yaxis.set_ticks_position("left")
    ax.spines["left"].set_position(("axes", -0.05))
    ax.spines["right"].set_visible(False)
    # labels
    ax.set_ylim([0, 1])
    return ax