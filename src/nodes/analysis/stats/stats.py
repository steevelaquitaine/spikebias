# # 1 hour minute
# fisher_exact(df.values[:25, :])
from scipy.stats import hypergeom
import numpy as np
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
from scipy.stats import ttest_1samp, ttest_ind
import pandas as pd

def hypergeom_probability(observations):
    row_marginals = np.sum(observations, axis=1)
    col_marginals = np.sum(observations, axis=0)
    N = np.sum(observations)
    p = 1
    for i in range(len(observations) - 1):
        for j in range(len(observations[i]) - 1):
            p *= hypergeom.pmf(
                observations[i][j],
                np.sum(col_marginals[j:]),
                col_marginals[j],
                row_marginals[i],
            )
            row_marginals[i] -= observations[i][j]
        col_marginals -= observations[i]
    return p


def multi_hypergeom_sample(m, colors):
    remaining = np.cumsum(colors[::-1])[::-1]
    result = np.zeros(len(colors), dtype=int)
    for i in range(len(colors) - 1):
        if m < 1:
            break
        result[i] = np.random.hypergeometric(colors[i], remaining[i + 1], m)
        m -= result[i]
    result[-1] = m
    return result


def sample_once(observations):
    row_marginals = np.sum(observations, axis=1)
    col_marginals = np.sum(observations, axis=0)
    sample = np.copy(observations)
    for i in range(len(row_marginals) - 1):
        sample[i] = multi_hypergeom_sample(row_marginals[i], col_marginals)
        col_marginals -= sample[i]
    sample[len(sample) - 1] = col_marginals
    return sample


def fisher_monte_carlo(observations, num_simulations=10 * 1000):
    p_obs = hypergeom_probability(observations)
    hits = 0
    for _ in range(num_simulations):
        sample = sample_once(observations)
        p_sample = hypergeom_probability(sample)
        if p_sample <= p_obs:
            hits += 1
    return hits / num_simulations


def get_ttest_quality(df_all, sorter):

    # set groups
    g_class = "mixed: good + overmerger + oversplitter"
    b_class = "mixed: overmerger + oversplitter"

    # get group data
    g_cap = df_all[(df_all["Unit class"] == g_class) & (df_all.Sorter == sorter)][
        "Capacity"
    ].values
    b_cap = df_all[(df_all["Unit class"] == b_class) & (df_all.Sorter == sorter)][
        "Capacity"
    ].values

    # run test
    t, p = ttest_ind(
        g_cap,
        b_cap,
    )
    dof = len(b_cap)-1
    print(f"""{sorter}: t({dof})={t}, p={p}""")
    
    
def get_ttest_good_vs_gt(df_all, sorter):

    # set groups
    g_class = "mixed: good + overmerger + oversplitter"

    # get group data
    g_cap = df_all[(df_all["Unit class"] == g_class) & (df_all.Sorter == sorter)][
        "Capacity"
    ].values
    gt_cap = df_all[(df_all["Unit class"] == "ground_truth") & (df_all.Sorter == "ground_truth")][
        "Capacity"
    ].values

    # run test
    t, p = ttest_ind(
        g_cap,
        gt_cap,
    )
    dof = len(gt_cap)-1
    print(f"""{sorter}: t({dof})={t}, p={p}""")
    
    
def get_one_samp_ttest_gt_vs_rand_subset(df_all: pd.DataFrame, sorter2: str):
    """run a one sample t-test for random subset vs ground truth
    """
    
    # get group data    
    cap1 = np.mean(df_all[(df_all["Sorter"] == "ground_truth") & (df_all["Sampling scheme"] == "random")][
        "Capacity"
    ].values)
    
    cap2 = df_all[(df_all["Sorter"] == sorter2) & (df_all["Sampling scheme"] == "random")][
        "Capacity"
    ].values

    # run test
    t, p = ttest_1samp(
        cap2,
        popmean=cap1,
    )
    dof = len(cap2)-1
    print(f"""Entire ground truth vs {sorter2}: t({dof})={t}, p={p}""")
    
    
def get_one_samp_ttest_gt_vs_biased_subset(df_all: pd.DataFrame, sorter2: str):
    """run a one sample t-test for biased subset vs ground truth
    """
    
    # get group data    
    cap1 = np.mean(df_all[(df_all["Sorter"] == "ground_truth") & (df_all["Sampling scheme"] == "random")][
        "Capacity"
    ].values)
    print(cap1)
    
    cap_bias = df_all[(df_all["Sorter"] == sorter2) & (df_all["Sampling scheme"] == "biased")][
        "Capacity"
    ].values

    # run test
    t, p = ttest_1samp(
        cap_bias,
        popmean=cap1,
    )
    dof = len(cap_bias)-1
    print(f"""Entire ground truth vs {sorter2}: t({dof})={t}, p={p}""")
    

def get_ttest_gt_vs_rand_subset(df_all: pd.DataFrame, sorter2: str):
    """run a one sample t-test for random subset vs ground truth
    """
    
    # get group data    
    cap1 = df_all[(df_all["Sorter"] == "ground_truth") & (df_all["Sampling scheme"] == "random")][
        "Capacity"
    ].values
    
    cap2 = df_all[(df_all["Sorter"] == sorter2) & (df_all["Sampling scheme"] == "random")][
        "Capacity"
    ].values

    # run test
    t, p = ttest_ind(
        cap2,
        cap1,
    )
    dof = len(cap2)-1
    print(f"""Entire ground truth vs {sorter2}: t({dof})={t}, p={p}""")
    
    
def get_ttest_gt_vs_biased_subset(df_all: pd.DataFrame, sorter2: str):
    """run a one sample t-test for biased subset vs ground truth
    """
    
    # get group data    
    cap1 = df_all[(df_all["Sorter"] == "ground_truth") & (df_all["Sampling scheme"] == "random")][
        "Capacity"
    ].values
    
    cap_bias = df_all[(df_all["Sorter"] == sorter2) & (df_all["Sampling scheme"] == "biased")][
        "Capacity"
    ].values

    # run test
    t, p = ttest_ind(
        cap_bias,
        cap1,
    )
    dof = len(cap_bias)-1
    print(f"""Entire ground truth vs {sorter2}: t({dof})={t}, p={p}""")
    
    
# by sorter -------------

def get_ttest_gt_vs_sorter(df_all: pd.DataFrame, sorter: str):
    """run a one sample t-test for random subset vs ground truth
    """
    
    # get group data    
    cap1 = df_all[df_all["Sorter"] == "ground_truth"]["Capacity"].values
    cap2 = df_all[df_all["Sorter"] == sorter]["Capacity"].values

    # run test
    t, p = ttest_ind(cap2, cap1,)
    dof = len(cap2)-1
    print(f"""Entire ground truth vs {sorter}: t({dof})={t}, p={p}""")