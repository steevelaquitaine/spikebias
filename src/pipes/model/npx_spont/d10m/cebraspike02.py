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
    python3.9 -c "from src.pipes.model.npx_spont.d10m.cebraspike import run; run()"
    
TODO:
- standardize datasets: 
    - quality table to lower case.
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
LAYERS = ["l4", "l5", "l6"]
EXP_TRAIN = "E"
EXP_TEST = "NS"
N_SPIKES = 50
N_SITES = 384
LOAD_WE_IF_EXISTS = True        # load waveform extractors for single units with pca
LOAD_AMP_IF_EXISTS = True       # load spike amplitudes
LOAD_QM_IF_EXISTS = False       # load quality metrics       
METRIC_NAMES = [
    "amplitude_cutoff",    
    "amplitude_cv",
    "snr",
    "isi_violations_ratio",
    "firing_range",
    "sd_ratio",    
    "silhouette"
]

# SETUP PIPE PARAMETERS ********************

torch.manual_seed(0)    # reproducibility
TRAIN = False           # train the model or load if exists
SHOW_DIMS = [0, 1, 2]   # select dimensions to plot
MIXED_MODEL = False     # use quality metrics as auxiliary labels
MODEL_CFG = {           # model parameters
    "model_architecture": "offset10-model", # receptive field size
    "distance": "cosine",
    "batch_size": 2048, # full batch is not implemented yet in this combination
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
                metric_names: list=["snr"], 
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
        continuous_qmetrics=True,
        metric_names=metric_names,
        qmetrics_in_dataset=True,
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
        continuous_qmetrics=True,
        metric_names=metric_names,
        qmetrics_in_dataset=True,
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
                                               metric_names=METRIC_NAMES,
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