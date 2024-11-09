"""_summary_

usage:

    sbatch cluster/figures/supp/plot_power_vs_sfreq.sbatch

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

N_CONTACTS = 384

# NPX-10KHz (1 col)
EXPERIMENT_10KHz = "silico_neuropixels"
SIMULATION_DATE_10KHz = "2023_06_26"
data_conf_10KHz, param_conf_10KHz = get_config(EXPERIMENT_10KHz, SIMULATION_DATE_10KHz).values() # confs
RAW_RECORDING_PATH_10KHz = data_conf_10KHz["recording"]["output"]
PREPRO_PATH_10KHz = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/preprocessed/2023_06_26/traces"
POWER_PATH_10KHz = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/realism/power/2023_06_26/"
SAMP_FREQ_10KHz = 10000


# NPX-40KHz (1 col)
EXPERIMENT_40KHz = "silico_neuropixels"
SIMULATION_DATE_40KHz = "2023_10_18"
data_conf_40KHz, param_conf_40KHz = get_config(EXPERIMENT_40KHz, SIMULATION_DATE_40KHz).values() # confs
RAW_RECORDING_PATH_40KHz = data_conf_40KHz["recording"]["output"]
PREPRO_PATH_40KHz = data_conf_40KHz["preprocessing"]["output"]["trace_file_path"]
POWER_PATH_40KHz = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/realism/power/2023_10_18/"
SAMP_FREQ_40KHz = 40000

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
raw_10KHz = si.load_extractor(RAW_RECORDING_PATH_10KHz)

# preprocess (1h30 min, uncomment to run) or load preprocessed
if LOAD_PREPROCESSED:
    prepro_10KHz = si.load_extractor(PREPRO_PATH_10KHz)
else:
    prepro_10KHz = amplitude.preprocess_silico(RAW_RECORDING_PATH_10KHz, PREPRO_PATH_10KHz, freq_min=300, freq_max=4999)

# get power spectrum
power_10KHz, freq_10KHz = power.get_power(raw_10KHz, N_CONTACTS, SAMP_FREQ_10KHz)
prepro_power_10KHz, _ = power.get_power(prepro_10KHz, N_CONTACTS, SAMP_FREQ_10KHz)

# save
if not os.path.isdir(POWER_PATH_10KHz):
    os.makedirs(POWER_PATH_10KHz)
np.save(POWER_PATH_10KHz+"power_raw.npy", power_10KHz)       
np.save(POWER_PATH_10KHz+"freq.npy", freq_10KHz)
np.save(POWER_PATH_10KHz+"power_prepro.npy", prepro_power_10KHz)


# 40KHz --------------

# load raw
raw_40KHz = si.load_extractor(RAW_RECORDING_PATH_40KHz)

# preprocess (1h30 min, uncomment to run) or load preprocessed
if LOAD_PREPROCESSED:
    prepro_40KHz = si.load_extractor(PREPRO_PATH_40KHz)
else:
    prepro_40KHz = amplitude.preprocess_silico(RAW_RECORDING_PATH_40KHz, PREPRO_PATH_40KHz, freq_min=300, freq_max=4999)


# get power spectrum
power_40KHz, freq_40KHz = power.get_power(raw_40KHz, N_CONTACTS, SAMP_FREQ_40KHz)
prepro_power_40KHz, _ = power.get_power(prepro_40KHz, N_CONTACTS, SAMP_FREQ_40KHz)

# save
if not os.path.isdir(POWER_PATH_40KHz):
    os.makedirs(POWER_PATH_40KHz)
np.save(POWER_PATH_40KHz+"power_raw.npy", power_40KHz)       
np.save(POWER_PATH_40KHz+"freq.npy", freq_40KHz)
np.save(POWER_PATH_40KHz+"power_prepro.npy", prepro_power_40KHz)



# PLOT 

# takes 2 mins

fig, axes = plt.subplots(1,2,figsize=(10,4))

# pick first the first frequencies
# amplitude to divide by
NORM_WIND = np.arange(0,5,1)

# raw --------------

# 10 KHz
psd, _, freq = downsample_power_data(POWER_PATH_10KHz)
psd_mean = np.mean(psd, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[0].plot(freq, norm_psd, color=[1,0,0]);
del psd; del freq

# 40 KHz
psd, _, freq = downsample_power_data(POWER_PATH_40KHz)
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
_, psd_pre, freq = downsample_power_data(POWER_PATH_10KHz)
psd_mean = np.mean(psd_pre, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[1].plot(freq, norm_psd, color=[1,0,0], label="npx 10KHz");
del psd_pre; del freq

# 40 KHz
_, psd_pre, freq = downsample_power_data(POWER_PATH_40KHz)
psd_mean = np.mean(psd_pre, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[1].plot(freq, norm_psd, color=[0.3,0.3,0.3], label="npx 40KHz");
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
plt.savefig("figures/4_controls/pdf/power_vs_sfreq.pdf")
plt.savefig("figures/4_controls/svg/power_vs_sfreq.svg")