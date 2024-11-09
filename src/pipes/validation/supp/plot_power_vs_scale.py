"""_summary_

usage:

    sbatch cluster/figures/supp/plot_power_vs_scale.sbatch

Returns:
    _type_: _description_
"""


import os
import numpy as np
import scipy
import spikeinterface.extractors as se 
from matplotlib import pyplot as plt
import matplotlib
import spikeinterface as si
import spikeinterface.preprocessing as spre

# move to PROJECT PATH
PROJ_PATH = "/gpfs/bbp.cscs.ch/project/proj68/home/laquitai/sfn_2023/"
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes.study import power
from src.nodes.study import amplitude

matplotlib.rcParams['agg.path.chunksize'] = 10000

SAMP_FREQ = 40000
N_CONTACTS = 384

# NPX 1col
EXPERIMENT_1col = "silico_neuropixels"
SIMULATION_DATE_1col = "2023_10_18"
data_conf_1col, param_conf_1col = get_config(EXPERIMENT_1col, SIMULATION_DATE_1col).values() # confs
RAW_RECORDING_PATH_1col = data_conf_1col["recording"]["output"]
PREPRO_PATH_1col = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/preprocessed/2023_10_18/traces"
POWER_PATH_1col = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/realism/power/2023_10_18/"


# NPX 7 cols
EXPERIMENT_7cols = "silico_neuropixels"
SIMULATION_DATE_7cols = "2023_08_17"
data_conf_7cols, param_conf_7cols = get_config(EXPERIMENT_7cols, SIMULATION_DATE_7cols).values() # confs
RAW_RECORDING_PATH_7cols = data_conf_7cols["recording"]["output"]
PREPRO_PATH_7cols = data_conf_7cols["preprocessing"]["output"]["trace_file_path"]
POWER_PATH_7cols = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/realism/power/2023_08_17/"


# setup pipeline
LOAD_PREPROCESSED = False



def downsample_power_data(data_path):
    psd = np.load(data_path+"power_raw.npy")
    psd = psd[:,::100]
    psd_pre = np.load(data_path+"power_prepro.npy")
    psd_pre = psd_pre[:,::100]
    freq = np.load(data_path+"freq.npy")
    if np.ndim(freq)>1:
        freq = freq[0,::100]
    else:
        freq = freq[::100]
    return psd, psd_pre, freq



# load raw
raw_1col = si.load_extractor(RAW_RECORDING_PATH_1col)

# preprocess (1h30 min, uncomment to run) or load preprocessed
if LOAD_PREPROCESSED:
    prepro_1col = si.load_extractor(PREPRO_PATH_1col)
else:
    prepro_1col = amplitude.preprocess_silico(RAW_RECORDING_PATH_1col, PREPRO_PATH_1col, freq_min=300, freq_max=4999)

# get power spectrum
power_1col, freq_1col = power.get_power(raw_1col, N_CONTACTS, SAMP_FREQ)
prepro_power_1col, _ = power.get_power(prepro_1col, N_CONTACTS, SAMP_FREQ)

# save
if not os.path.isdir(POWER_PATH_1col):
    os.makedirs(POWER_PATH_1col)
np.save(POWER_PATH_1col+"power_raw.npy", power_1col)       
np.save(POWER_PATH_1col+"freq.npy", freq_1col)
np.save(POWER_PATH_1col+"power_prepro.npy", prepro_power_1col)


# 7cols --------------

# load raw
raw_7cols = si.load_extractor(RAW_RECORDING_PATH_7cols)

# preprocess (1h30 min, uncomment to run) or load preprocessed
if LOAD_PREPROCESSED:
    prepro_7cols = si.load_extractor(PREPRO_PATH_7cols)
else:
    prepro_7cols = amplitude.preprocess_silico(RAW_RECORDING_PATH_7cols, PREPRO_PATH_7cols, freq_min=300, freq_max=4999)


# get power spectrum
power_7cols, freq_7cols = power.get_power(raw_7cols, N_CONTACTS, SAMP_FREQ)
prepro_power_7cols, _ = power.get_power(prepro_7cols, N_CONTACTS, SAMP_FREQ)

# save
if not os.path.isdir(POWER_PATH_7cols):
    os.makedirs(POWER_PATH_7cols)
np.save(POWER_PATH_7cols+"power_raw.npy", power_7cols)       
np.save(POWER_PATH_7cols+"freq.npy", freq_7cols)
np.save(POWER_PATH_7cols+"power_prepro.npy", prepro_power_7cols)



# PLOT 

# takes 2 mins

fig, axes = plt.subplots(1,2,figsize=(10,4))

# pick first the first frequencies
# amplitude to divide by
NORM_WIND = np.arange(0,5,1)

# raw --------------

# 10 KHz
psd, _, freq = downsample_power_data(POWER_PATH_1col)
psd_mean = np.mean(psd, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[0].plot(freq, norm_psd, color=[1,0,0]);
del psd; del freq

# 40 KHz
psd, _, freq = downsample_power_data(POWER_PATH_7cols)
psd_mean = np.mean(psd, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[0].plot(freq, norm_psd, color=[0.3,0.3,0.3]);
del psd; del freq

# legend
axes[0].set_xlabel("Log frequency (Hz)");
axes[0].set_ylabel("Log normalized power (AU)");
axes[0].set_yscale("log");
axes[0].set_xscale("log");
axes[0].spines[['right', 'top']].set_visible(False);
axes[0].set_ylim([1e-9, 1e5]);

# show minor ticks
axes[0].tick_params(which='both', width=1)
locmaj = matplotlib.ticker.LogLocator(base=10,numticks=12) 
axes[0].xaxis.set_major_locator(locmaj)    
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=12)
axes[0].xaxis.set_minor_locator(locmin)
axes[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

# preprocessed --------------
# 10 KHz
_, psd_pre, freq = downsample_power_data(POWER_PATH_1col)
psd_mean = np.mean(psd_pre, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[1].plot(freq, norm_psd, color=[1,0,0], label="npx 1 col");
del psd_pre; del freq

# 40 KHz
_, psd_pre, freq = downsample_power_data(POWER_PATH_7cols)
psd_mean = np.mean(psd_pre, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[1].plot(freq, norm_psd, color=[0.3,0.3,0.3], label="npx 7 cols");
del psd_pre; del freq

# legend
axes[1].set_xlabel("Log frequency (Hz)");
axes[1].set_ylabel("Log normalized power (AU)");
axes[1].set_yscale("log");
axes[1].set_xscale("log");
axes[1].spines[['right', 'top']].set_visible(False);
axes[1].legend(frameon=False, loc='center left', bbox_to_anchor=(1, 0.5));
axes[1].set_ylim([1e-7, 1e5]);

# show minor ticks
axes[1].tick_params(which='both', width=1)
locmaj = matplotlib.ticker.LogLocator(base=10, numticks=12)
axes[1].xaxis.set_major_locator(locmaj)
locmin = matplotlib.ticker.LogLocator(base=10.0, subs=(0.2,0.4,0.6,0.8), numticks=12)
axes[1].xaxis.set_minor_locator(locmin)
axes[1].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.tight_layout();

# save figures
plt.savefig("figures/4_controls/pdf/scale_wise_power.pdf")
plt.savefig("figures/4_controls/svg/scale_wise_power.svg")