"""Pipeline that plots pfr-wise power spectrum figure

usage:

    sbatch cluster/figures/supp/plot_power_vs_pfr.sbatch

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
SAMP_FREQ = 40000

# NPX-pfr05 (1 col)
EXPERIMENT_pfr05 = "silico_neuropixels"
SIMULATION_DATE_pfr05 = "2023_09_12"
data_conf_pfr05, param_conf_pfr05 = get_config(EXPERIMENT_pfr05, SIMULATION_DATE_pfr05).values()
RAW_RECORDING_PATH_pfr05 = data_conf_pfr05["recording"]["output"]
PREPRO_PATH_pfr05 = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/preprocessed/2023_09_12/traces"
POWER_PATH_pfr05 = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/realism/power/2023_09_12/"

# NPX-pfr03 (1 col)
EXPERIMENT_pfr03 = "silico_neuropixels"
SIMULATION_DATE_pfr03 = "2023_10_18"
data_conf_pfr03, param_conf_pfr03 = get_config(EXPERIMENT_pfr03, SIMULATION_DATE_pfr03).values() # confs
RAW_RECORDING_PATH_pfr03 = data_conf_pfr03["recording"]["output"]
PREPRO_PATH_pfr03 = data_conf_pfr03["preprocessing"]["output"]["trace_file_path"]
POWER_PATH_pfr03 = "/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/realism/power/2023_10_18/"

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
raw_pfr05 = si.load_extractor(RAW_RECORDING_PATH_pfr05)

# preprocess (1h30 min, uncomment to run) or load preprocessed
if LOAD_PREPROCESSED:
    prepro_pfr05 = si.load_extractor(PREPRO_PATH_pfr05)
else:
    prepro_pfr05 = amplitude.preprocess_silico(RAW_RECORDING_PATH_pfr05, PREPRO_PATH_pfr05, freq_min=300, freq_max=4999)

# get power spectrum
power_pfr05, freq_pfr05 = power.get_power(raw_pfr05, N_CONTACTS, SAMP_FREQ)
prepro_power_pfr05, _ = power.get_power(prepro_pfr05, N_CONTACTS, SAMP_FREQ)

# save
if not os.path.isdir(POWER_PATH_pfr05):
    os.makedirs(POWER_PATH_pfr05)
np.save(POWER_PATH_pfr05+"power_raw.npy", power_pfr05)       
np.save(POWER_PATH_pfr05+"freq.npy", freq_pfr05)
np.save(POWER_PATH_pfr05+"power_prepro.npy", prepro_power_pfr05)


# 40KHz --------------

# load raw
raw_pfr03 = si.load_extractor(RAW_RECORDING_PATH_pfr03)

# preprocess (1h30 min, uncomment to run) or load preprocessed
if LOAD_PREPROCESSED:
    prepro_pfr03 = si.load_extractor(PREPRO_PATH_pfr03)
else:
    prepro_pfr03 = amplitude.preprocess_silico(RAW_RECORDING_PATH_pfr03, PREPRO_PATH_pfr03, freq_min=300, freq_max=4999)


# get power spectrum
power_pfr03, freq_pfr03 = power.get_power(raw_pfr03, N_CONTACTS, SAMP_FREQ)
prepro_power_pfr03, _ = power.get_power(prepro_pfr03, N_CONTACTS, SAMP_FREQ)

# save
if not os.path.isdir(POWER_PATH_pfr03):
    os.makedirs(POWER_PATH_pfr03)
np.save(POWER_PATH_pfr03+"power_raw.npy", power_pfr03)       
np.save(POWER_PATH_pfr03+"freq.npy", freq_pfr03)
np.save(POWER_PATH_pfr03+"power_prepro.npy", prepro_power_pfr03)



# PLOT 

# takes 2 mins

fig, axes = plt.subplots(1,2,figsize=(10,4))

# pick first the first frequencies
# amplitude to divide by
NORM_WIND = np.arange(0,5,1)

# raw --------------

# 10 KHz
psd, _, freq = downsample_power_data(POWER_PATH_pfr05)
psd_mean = np.mean(psd, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[0].plot(freq, norm_psd, color=[1,0,0]);
del psd; del freq

# 40 KHz
psd, _, freq = downsample_power_data(POWER_PATH_pfr03)
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
# pfr=0.5
_, psd_pre, freq = downsample_power_data(POWER_PATH_pfr05)
psd_mean = np.mean(psd_pre, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[1].plot(freq, norm_psd, color=[1,0,0], label="npx pfr=0.5");
del psd_pre; del freq

# pfr=0.3
_, psd_pre, freq = downsample_power_data(POWER_PATH_pfr03)
psd_mean = np.mean(psd_pre, axis=0)
norm_psd = psd_mean/np.mean(psd_mean[NORM_WIND])
axes[1].plot(freq, norm_psd, color=[0.3,0.3,0.3], label="npx pfr=0.3");
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
plt.savefig("figures/4_controls/pdf/pfr_wise_power.pdf")
plt.savefig("figures/4_controls/svg/pfr_wise_power.svg")