"""sort marques silico with Kilosort 3.0 (spikeinterface==0.100.5)
takes 3h

  author: steeve.laquitaine@epfl.ch
    date: 15.01.2023
modified: 29.04.2024

usage: 

    sbatch cluster/sorting/marques_silico/wavelet/sort_ks3.sbatch

We set Kilosort3 to do no preprocessing as we apply wavelet filtering
before.

Notes:
    we set scaleproc=200 explicitly (default value when using preprocessing)
    AssertionError: When skip_kilosort_preprocessing=True scaleproc must explicitly given
    
    remove drift correction: do_correction=False, else crashes

"""
import logging
import logging.config
import shutil
from time import time
import spikeinterface as si
import spikeinterface.sorters as ss
import yaml
from src.nodes.utils import get_config

# SET PARAMETERS
sorter_params = {
    "detect_threshold": 138, # 138,  # 12, # 6, # alters detection on a per spike basis and is applied to the voltage trace (default=6)
    "projection_threshold": [9, 9], # [9, 9],  # threshold for detected projected spikes (energy) during two passes
    "preclust_threshold": 8,
    "car": True,
    "minFR": 0.2, # Minimum spike rate (Hz), if a cluster falls below this for too long it gets removed"
    "minfr_goodchannels": 0.2,  # Minimum firing rate on a 'good' channel"
    "nblocks": 5,
    "sig": 20,
    "freq_min": 300,
    "sigmaMask": 30,
    "lam": 20.0,
    "nPCs": 3,
    "ntbuff": 64,
    "nfilt_factor": 4,
    "do_correction": False, # True,
    "NT": 65792, # None,
    "AUCsplit": 0.8, # Threshold on the area under the curve (AUC) criterion for performing a split in the final step
    "wave_length": 61,
    "keep_good_only": False,
    "skip_kilosort_preprocessing": False, # no preprocessing (I commented filtering commands in KS3 gpufilter.m)
    "scaleproc": None, # 200 #is the scaling aaplied after whitening during preprocessing by default (when "None")
    "save_rez_to_mat": False,
    "delete_tmp_files": ("matlab_files",),
    "delete_recording_dat": False,
}

# SET PARAMETERS
# sorter_params = {
#     'detect_threshold': 138, # 6      # spike detection thresh.
#     'projection_threshold': [9, 9], # template-projected detection thresh.
#     'preclust_threshold': 8,
#     'car': True,
#     'minFR': 0.2,
#     'minfr_goodchannels': 0.2,
#     'nblocks': 5,
#     'sig': 20,
#     'freq_min': 300,                # high-pass filter cutoff
#     'sigmaMask': 30,
#     'nPCs': 3,
#     'ntbuff': 64,
#     'nfilt_factor': 4,
#     'do_correction': True,
#     'NT': 65792, # None,            # Batch size
#     'wave_length': 61,
#     'keep_good_only': False
#     }

# SETUP CONFIG
data_conf, _ = get_config("silico_neuropixels", "concatenated").values()

# setup logging
with open("conf/logging.yml", "r", encoding="utf-8") as logging_conf:
    LOG_CONF = yaml.load(logging_conf, Loader=yaml.FullLoader)
logging.config.dictConfig(LOG_CONF)
logger = logging.getLogger("root")

# SET READ and write PATHS
# preprocessed trace (detrending and wavelet-filtering)
PREPRO_PATH = data_conf["preprocessing"]["output"]['wavelet_trace_file_path']  # read
KS3_PACKAGE_PATH = data_conf['sorting']['sorters']['kilosort3']['input']  # read
KS3_SORTING_PATH = data_conf['sorting']['sorters']['kilosort3']['wavelet']['output']  # write
KS3_OUTPUT_PATH = data_conf['sorting']['sorters']['kilosort3']['wavelet']['ks3_output']  # write

# test sample butterworth
#PREPRO_PATH = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/preprocessed/trace_sample" # butterworth
#KS3_SORTING_PATH = '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/SortingKS3_sample/' # butter
#KS3_OUTPUT_PATH = '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/KS3_output_sample/' # butter

# test sample wavelet 
#PREPRO_PATH = "/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/preprocessed/trace_sample_wavelet"
#KS3_SORTING_PATH = '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/SortingKS3_sample_wvlt/'
#KS3_OUTPUT_PATH = '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/KS3_output_sample_wvlt/'

# SET KS3 environment variable
ss.Kilosort3Sorter.set_kilosort3_path(KS3_PACKAGE_PATH)

# get Spikeinterface Recording object
# enforce int16 dtype (speed)
t0 = time()
Recording = si.load_extractor(PREPRO_PATH)
assert Recording.dtype == "int16", f"RecordingExtractor should be int16 dtype, not {Recording.dtype}"

# make sure traces are float32 (not float64)
# to speed up sorting and reduce memory cost
logger.info("Done loading & converting Recording as float32 in: %s",
            round(time() - t0, 1))

# run sorting (default parameters)
# with Recording.dtype="int16", sorting with Kilosort 
# does not re-copy the Recording binary, which can take hours
# sorting is also at least 8X faster than with float64
t0 = time()
sorting_KS3 = ss.run_sorter(sorter_name='kilosort3',
                            recording=Recording,
                            remove_existing_folder=True,
                            output_folder=KS3_OUTPUT_PATH,
                            verbose=True,
                            **sorter_params)

# remove empty units (Samuel Garcia's advice)
sorting_KS3 = sorting_KS3.remove_empty_units()
logger.info("Done running kilosort3 in: %s", round(time() - t0, 1))

# write
shutil.rmtree(KS3_SORTING_PATH, ignore_errors=True)
sorting_KS3.save(folder=KS3_SORTING_PATH)
logger.info("Done saving kilosort3 in: %s", round(time() - t0, 1))