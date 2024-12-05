"""pipeline to fully curate single-units with cebraspike model:

Run parameters: 

- sorter: kilosort4
- training dataset: evoked regime (npx biophy)  
- test dataset: spontaneous regime (npx biophy) 
- recording duration: 10 minutes

Usage:

    # run as a module
    source /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/envs/cebraspike3/bin/activate
    cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/
    export PYTHONPATH=$(pwd)
    python3.9 -c "from src.pipes.model.npx_spont.d10m.cebraspike01 import run; run()"
    
TODO:

- standardize datasets:  
    - quality table to lower case.

* test only 1d waveforms:
    * no quality metrics
    * performance: pooled: {'precision': 0.6309523809523809, 'recall': 0.6309523809523809, 'f1_score': 0.6309523809523809}
    * train sample size: 408 units (L23, 4,5,6)
    * test sample size: 184 units (L23, L4,5,6)
    * n_spikes: 25
    * batch_size: 1024
    * should be my run 5
    
* test only 1d waveforms [RUNNING ....]:
    * no quality metrics
    * performance: pooled:
    * train sample size:
    * test sample size:
    * n_spikes: 10
    * batch_size: 1024
    
* test only 1d waveforms [RUNNING ....]:
    * no quality metrics
    * performance: pooled: {'precision': 0.6309523809523809, 'recall': 0.6309523809523809, 'f1_score': 0.6309523809523809}
    * train sample size: 408 units (L4,5,6)
    * test sample size: 184 units (L4,5,6)
    * n_spikes: 50
    * batch_size: 2048  
    
* Results:
    * with these metrics (7): 
        * metrics: ['amplitude_cutoff', 'amplitude_cv', 'snr', 'isi_violations_ratio',
        'firing_range', 'sd_ratio', 'silhouette']
    * performance: pooled: {'mae': 0.0, 'r2': -0.26620029455080974, 'accuracy': 0.6850828729281768, 'bal_accuracy': 0.6846465390279823, 'precision': 0.6551724137931034, 'recall': 0.6785714285714286, 'f1_score': 0.6666666666666666}

* with these metrics (11):
    * ['amplitude_cutoff', 'firing_range', 'firing_rate',
       'isi_violations_ratio', 'presence_ratio', 'rp_contamination',
       'rp_violations', 'sd_ratio', 'snr', 'silhouette', 'amplitude_cv']
    * performance: pooled: {'mae': 0.0, 'r2': -0.4439126165930287, 'accuracy': 0.6408839779005525, 'bal_accuracy': 0.6410162002945508, 'precision': 0.6067415730337079, 'recall': 0.6428571428571429, 'f1_score': 0.6242774566473989}       
    
* test without amplitude_cutoff (10):
    * ['firing_range', 'firing_rate',
       'isi_violations_ratio', 'presence_ratio', 'rp_contamination',
       'rp_violations', 'sd_ratio', 'snr', 'silhouette', 'amplitude_cv']
    * performance: pooled: pooled: {'mae': 0.0, 'r2': -0.37727049582719663, 'accuracy': 0.6574585635359116, 'bal_accuracy': 0.6485027000490918, 'precision': 0.6666666666666666, 'recall': 0.5238095238095238, 'f1_score': 0.5866666666666667}
    * amplitude_cutoff has lots of missing values and result in reducing training unit sample by half
        * but it seem to increase recall.
    
* with all qmetrics available w/o nan except heavy pca-based (20):
    * ['amplitude_cutoff', 'amplitude_cv_median', 'amplitude_cv_range',
       'amplitude_median', 'firing_range', 'firing_rate',
       'isi_violations_ratio', 'isi_violations_count', 'num_spikes',
       'presence_ratio', 'rp_contamination', 'rp_violations', 'sd_ratio',
       'sliding_rp_violation', 'snr', 'sync_spike_2', 'sync_spike_4',
       'sync_spike_8', 'silhouette', 'amplitude_cv']
    * breaks because several metrics are all nan (e.g., amplitude_median)

* test with all non-nan metrics, and amplitude_cutoff parametrized to produce no NaN (10):
    * ['amplitude_cutoff', 'firing_range', 'firing_rate',
       'isi_violations_ratio', 'presence_ratio', 'rp_contamination',
       'rp_violations', 'sd_ratio', 'snr', 'silhouette', 'amplitude_cv']
    * pooled: {'mae': 0.0, 'r2': -0.5327687776141381, 'accuracy': 0.6187845303867403, 'bal_accuracy': 0.5932744231713304, 'precision': 0.8, 'recall': 0.23809523809523808, 'f1_score': 0.3669724770642202}
        * takes 41 minutes
        
* test with all non-nan metrics, and amplitude_cutoff parametrized to just a few NaN (10):
    * ['amplitude_median', 'firing_range', 'firing_rate',
       'isi_violations_ratio', 'isi_violations_count', 'num_spikes',
       'presence_ratio', 'rp_contamination', 'rp_violations', 'sd_ratio',
       'snr', 'sync_spike_2', 'sync_spike_4', 'sync_spike_8',
       'amplitude_cutoff', 'amplitude_cv']
    * pooled: {'mae': 0.0, 'r2': -0.4519354011791852, 'accuracy': 0.6384180790960452, 'bal_accuracy': 0.6313765701102281, 'precision': 0.6417910447761194, 'recall': 0.5180722891566265, 'f1_score': 0.5733333333333334}
    * sample size = 179 units
    * takes 41 minute
        
        
* test with initially tested metrics subset, w/o amplitude_cutoff nor silhouette:
    * ["snr", "isi_violations_ratio", "isi_violations_count", 
    "amplitude_cv", "sd_ratio", "firing_rate", 
    "presence_ratio"]        
    * pooled: {'mae': 0.0, 'r2': -0.3550564555719191, 'accuracy': 0.6629834254143646, 'bal_accuracy': 0.6656234658811978, 'precision': 0.6210526315789474, 'recall': 0.7023809523809523, 'f1_score': 0.659217877094972}
    * train sample size: 408 units
    * test sample size: 184 units
    
* test with initially tested metrics subset, + silhouette:
    * ["snr", "isi_violations_ratio", "isi_violations_count",
    "amplitude_cv", "sd_ratio", "firing_rate", 
    "presence_ratio"]        
    * pooled: {'mae': 0.0, 'r2': -0.3550564555719191, 'accuracy': 0.6629834254143646, 'bal_accuracy': 0.6656234658811978, 'precision': 0.6210526315789474, 'recall': 0.7023809523809523, 'f1_score': 0.659217877094972}
    * train sample size: 408 units
    * test sample size: 184 units
    
"""

import os
import spikeinterface as si
import spikeinterface.core.template_tools as ttools
from spikeinterface import comparison
from spikeinterface.qualitymetrics import compute_quality_metrics as qm
from spikeinterface import qualitymetrics
import pandas as pd
from cebra import CEBRA
import cebra
import torch
import numpy as np
from matplotlib import pyplot as plt
import sklearn
import seaborn as sns
from sklearn import metrics
import cebra.models
import shutil
from spikeinterface.postprocessing import compute_principal_components
import multiprocessing
import pickle
import time
import yaml 

# move to project path
with open("./proj_cfg.yml", "r", encoding="utf-8") as proj_cfg:
    PROJ_PATH = yaml.load(proj_cfg, Loader=yaml.FullLoader)["proj_path"]
os.chdir(PROJ_PATH)

from src.nodes.utils import get_config
from src.nodes import utils
from src.nodes.utils import euclidean_distance
from src.nodes.models import utils as mutils
from src.nodes.models.utils import dataset as dutils
from src.nodes.models.models import CebraSpike1


# SETUP DATASET PATHS  *************************************

# npx spont. biophy.
cfg_ns, _ = get_config("silico_neuropixels", "concatenated").values()
KS4_ns_10m = cfg_ns["sorting"]["sorters"]["kilosort4"]["10m"][
    "output"
]  # sorting with KS4
GT_ns_10m = cfg_ns["ground_truth"]["10m"]["output"] # KS4 sorting
STUDY_ns = cfg_ns["postprocessing"]["waveform"]["sorted"]["study"]["kilosort4"][
    "10m"
]  # WaveformExtractor
STUDY_ns_su = '/gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/0_silico/neuropixels/concatenated_campaigns/postpro/realism/spike/sorted/study_ks4_10m_single_units'
REC_ns = cfg_ns["probe_wiring"]["full"]["output"]  # Wired

# npx evoked biophy.
cfg_ne, _ = get_config("silico_neuropixels", "stimulus").values()
KS4_ne_10m = cfg_ne["sorting"]["sorters"]["kilosort4"]["10m"]["output"]
GT_ne_10m = cfg_ne["ground_truth"]["10m"]["output"]
STUDY_ne = cfg_ne["postprocessing"]["waveform"]["sorted"]["study"]["kilosort4"][
    "10m"
]  # WaveformExtractor
REC_ne = cfg_ne["probe_wiring"]["full"]["output"]  # Wired
STUDY_ne_su = '/gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/postprocessing/biophy/4_spikesorting_stimulus_test_neuropixels_8-1-24__8slc_80f_360r_50t_200ms_1_smallest_fiber_gids/sorted/study_ks4_10m_single_units'

# job parameters
job_kwargs = dict(n_jobs=-1, progress_bar=True)

# pre-computed sorted unit quality
quality_path = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/analysis/sorting_quality/sorting_quality.csv"

# parameters

SORTER = "KS4"
LAYERS = ["l23", "l4", "l5", "l6"]
EXP_TRAIN = "E"
EXP_TEST = "NS"
N_SPIKES = 10
N_SITES = 384
LOAD_WE_IF_EXISTS = True        # load waveform extractors for single units with pca
LOAD_AMP_IF_EXISTS = True       # load spike amplitudes
CONTINUOUS_QMETRICS = False
QMETRICS_IN_DATASET = False
LOAD_QM_IF_EXISTS = False       # load quality metrics 

# SETUP PIPE PARAMETERS ********************

torch.manual_seed(0)    # reproducibility
TRAIN = True            # train the model or load if exists
SHOW_DIMS = [0, 1, 2]   # select dimensions to plot
MIXED_MODEL = False     # use quality metrics as auxiliary labels
MODEL_CFG = {           # model parameters
    "model_architecture": "offset10-model", # receptive field size
    "distance": "cosine",
    "batch_size": 1024, # full batch is not implemented yet in this combination
    "temperature_mode": "auto",
    "learning_rate": 0.001,
    "max_iterations": 10000,
    "conditional": "time_delta",
    "time_offsets": 10, # 10 timesteps (should be >= nb of receptve fields)
    "output_dimension": 10,
    "device": "cuda_if_available",
    "verbose": True,
}
MODEL_SAVE_PATH = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/analysis/sorting_quality/models/cebra/sf_20Khz/e2s_pm4_mixed_dataset_1dwave_qmall/"
RESULTS_SAVE_PATH = "/gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/analysis/sorting_quality/models/cebra/sf_20Khz/e2s_pm4_mixed_dataset_1dwave_qmall/results.pickle"


# SETUP PLOTS  ***************************
tight_layout_cfg = {"pad": 0.001}


def check_gpu():
    """check gpu specifications
    """
    # check GPU specs
    print("GPU specs:", multiprocessing.cpu_count())
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("__CUDNN VERSION:", torch.backends.cudnn.version())
        print("__Number CUDA Devices:", torch.cuda.device_count())
        print("__CUDA Device Name:", torch.cuda.get_device_name(0))
        print(
            "__CUDA Device Total Memory [GB]:",
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )
    # check CPU specs
    print("CPU specs:", multiprocessing.cpu_count())
    

def get_dataset(sorter: str, 
                exp_train: str, 
                exp_test: str, 
                sort_ns: str, 
                study_ns: str,
                study_ns_su: str, 
                sort_ne: str, 
                study_ne: str, 
                study_ne_su: str, 
                qpath: str, 
                num_spikes: int=50,
                n_sites: int = 384, 
                load_we_if_exists: bool = True,
                load_amp_if_exists: bool = True,
                load_qm_if_exists: bool = True, 
                job_kwargs: dict={"n_job": -1}):
    """load and format the datasets to model and decode

    Args:
        sort_ns (str): _description_
        study_ns (str): _description_
        study_ns_su (str): _description_
        sort_ne (str): _description_
        study_ne (str): _description_
        study_ne_su (str): _description_
        load_if_exists (bool): _description_
        n_sites (int): _description_
        qpath (str): path of pd.DataFrame of evaluated sorted units
        job_kwargs (dict): _description_

    Returns:
        _type_: _description_
        
    Note: 
    - Cebra requires a minimum of 10 units.
    """
    
    # load unit quality (sorting_quality.py pipe)
    quality = pd.read_csv(qpath)
    
    # (40s)for single units
    # note: adding PCA takes 3 hours (do once, then set load_if_exists=True)
    # - WaveformExtractor for training (evoked)
    WeNe = dutils.get_waveformExtractor_for_single_units(
        sort_ne, study_ne, study_ne_su, n_sites=n_sites, 
        load_if_exists=load_we_if_exists, job_kwargs=job_kwargs
    )
    # - WaveformExtractor for testing (spontaneous)
    WeNs = dutils.get_waveformExtractor_for_single_units(
        sort_ns, study_ns, study_ns_su, n_sites=n_sites, 
        load_if_exists=load_we_if_exists, job_kwargs=job_kwargs
    )    

    # training dataset (evoked regime by layer and pooled)
    data_lyr_ev = dutils.get_dataset_by_layer(
        sort_path=sort_ne,
        gt_path=GT_ne_10m,
        we=WeNe,
        qpath=quality_path,
        sorter=sorter,
        exp=exp_train,
        num_spike=num_spikes,
        interval_ms=3,
        downsample=1,
        continuous_qmetrics=CONTINUOUS_QMETRICS,
        qmetrics_in_dataset=QMETRICS_IN_DATASET,
        load_amp_if_exists=load_amp_if_exists,
        load_qm_if_exists=load_qm_if_exists, 
        wave_dim=1,
        layers=LAYERS
    )
    data_pooled_ev = dutils.get_dataset_pooled(
        data_lyr_ev, layers=LAYERS
    )

    # test dataset (spontaneous regime by layer and pooled)
    data_lyr_sp = dutils.get_dataset_by_layer(
        sort_path=sort_ns,
        gt_path=GT_ns_10m,
        we=WeNs,
        qpath=quality_path,
        sorter=sorter,
        exp=exp_test,
        num_spike=num_spikes,
        interval_ms=3,
        downsample=2,
        continuous_qmetrics=CONTINUOUS_QMETRICS,
        qmetrics_in_dataset=QMETRICS_IN_DATASET,
        load_amp_if_exists=load_amp_if_exists,
        load_qm_if_exists=load_qm_if_exists,        
        wave_dim=1,
        layers=LAYERS
    )
    data_pooled_sp = dutils.get_dataset_pooled(
        data_lyr_sp, layers=LAYERS
    )
    return WeNs, WeNe, quality, data_lyr_sp, data_pooled_sp, data_lyr_ev, data_pooled_ev


def run():
    """entry point that runs pipeline
    
    Args:
        quality_path (str): path of evaluated sorted single-units
        - produced by "sorting_quality.py" pipeline
        sorter: "KS4", "KS3", "KS2.5", "KS2", "KS", "HS"
        - the sorter labels available in "quality_path"
        exp_train: "NS", "E", "DS"
        - for npx biophy spont, npx evoked, dense 
        biophy spontaneous
        
        exp_test: same 
    
    """
    # track time
    t0 = time.time()
    
    # print specifications
    check_gpu()
    
    # load and format train and test datasets
    (WeNs, WeNe, unit_quality, data_lyr_sp, data_pooled_sp, 
    data_lyr_ev, data_pooled_ev) = get_dataset(sorter=SORTER, 
                                               exp_train=EXP_TRAIN, 
                                               exp_test=EXP_TEST, 
                                               sort_ns=KS4_ns_10m, 
                                               study_ns=STUDY_ns, 
                                               study_ns_su=STUDY_ns_su, 
                                               sort_ne=KS4_ne_10m, 
                                               study_ne=STUDY_ne, 
                                               study_ne_su=STUDY_ne_su, 
                                               qpath=quality_path,
                                               num_spikes=N_SPIKES,
                                               load_we_if_exists=LOAD_WE_IF_EXISTS,
                                               load_amp_if_exists=LOAD_AMP_IF_EXISTS,
                                               load_qm_if_exists=LOAD_QM_IF_EXISTS,
                                               n_sites=N_SITES,
                                               job_kwargs=job_kwargs)
    
    # (2h/load: 10s) train, test and evaluate
    results = CebraSpike1.train_test_eval(
        MODEL_CFG,
        TRAIN,
        MODEL_SAVE_PATH,
        data_lyr_ev,  # train
        data_pooled_sp, # test
        data_pooled_ev, # train
        SHOW_DIMS,
        tight_layout_cfg,
        mixed_model=MIXED_MODEL, # qmetrics are not auxiliary labels
        layers=LAYERS
    )
    
    # save results
    with open(RESULTS_SAVE_PATH, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print(f"All completed in {np.round(time.time()-t0,2)}")
    