
"""Pipeline to plot pfr-wise sorting accuracy

usage:

    sbatch cluster/figures/supp/plot_accuracy_vs_pfr.sbatch

"""

### SETUP  ------------

# SETUP PACKAGES 
import os
import numpy as np
from matplotlib import pyplot as plt

# SET PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/sfn_2023"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.postpro.accuracy import get_sorting_accuracies

# npx384 - 1 col
EXPERIMENT_pfr03 = "silico_neuropixels"
SIMULATION_DATE_pfr03 = "2023_10_18"
data_conf_pfr03, param_conf_pfr03 = get_config(EXPERIMENT_pfr03, SIMULATION_DATE_pfr03).values()
GT_SORTING_PATH_pfr03 = data_conf_pfr03["ground_truth"]["full"]["output"]
KS3_SORTING_PATH_pfr03 = data_conf_pfr03["sorting"]["sorters"]["kilosort3"]["output"]

# npx-384 - 7 cols
EXPERIMENT_pfr05 = "silico_neuropixels"
SIMULATION_DATE_pfr05 = "2023_09_12"
data_conf_pfr05, _ = get_config(EXPERIMENT_pfr05, SIMULATION_DATE_pfr05).values()
KS3_SORTING_PATH_pfr05 = data_conf_pfr05["sorting"]["sorters"]["kilosort3"]["output"]
GT_SORTING_PATH_pfr05 = data_conf_pfr05["ground_truth"]["full"]["output"]


### COMPUTE  ------------

# takes 44 min

# get 1 col accuracies
accuracies_pfr03 = get_sorting_accuracies(GT_SORTING_PATH_pfr03, KS3_SORTING_PATH_pfr03)
acc_array_pfr03 = np.array(accuracies_pfr03)

# get 7 cols accuracies
accuracies_pfr05 = get_sorting_accuracies(GT_SORTING_PATH_pfr05, KS3_SORTING_PATH_pfr05)
acc_array_pfr05 = np.array(accuracies_pfr05)


### PLOT ----------

MARKERSIZE = 5

# count units
n_units = max([len(accuracies_pfr03), len(accuracies_pfr05)])

# plot
fig, axis = plt.subplots(1,1,figsize=(5,3))
axis.plot(acc_array_pfr03, label=f"npx (pfr=0.3), n={len(acc_array_pfr03)})", marker="o", markerfacecolor="w", markeredgecolor=[1,0,0], color=[1,0,0], linestyle="-", markersize=MARKERSIZE, markeredgewidth=0.4, linewidth=1);
axis.plot(acc_array_pfr05, label=f"npx (pfr=0.5), n={len(acc_array_pfr05)})", marker="o", markerfacecolor="w", markeredgecolor=[.3,.3,.3], color=[.3,.3,.3], linestyle="-", markersize=MARKERSIZE, markeredgewidth=0.4, linewidth=1);

# add legend
axis.spines[["right", "top"]].set_visible(False);
axis.set_ylabel("Sorting accuracy (ratio)", fontsize=9);
axis.set_xlabel("True units ranked by sorting accuracy (id)", fontsize=9);
axis.set_xlim([-5, n_units]);
axis.legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=9);

# save figures
plt.savefig("figures/4_controls/pdf/pfr_wise_accuracy.pdf")
plt.savefig("figures/4_controls/svg/pfr_wise_accuracy.svg")



### TOTAL ACCURACY -----------

# Good detection threshold
DET_THRESH = 0.8

print("Well detected units for npx384, 1 col (id):", accuracies_pfr03[accuracies_pfr03 >= DET_THRESH].index.tolist())
print("Number of well detected units:", sum(acc_array_pfr03 >= DET_THRESH))
print("accuracy:", len(acc_array_pfr03[acc_array_pfr03 >= DET_THRESH])/len(acc_array_pfr03))

print("Well detected units for npx384, 7 cols (id):", accuracies_pfr05[accuracies_pfr05 >= DET_THRESH].index.tolist())
print("Number of well detected units:", sum(acc_array_pfr05 >= DET_THRESH))
print("accuracy:", len(acc_array_pfr05[acc_array_pfr05 >= DET_THRESH])/len(acc_array_pfr05))