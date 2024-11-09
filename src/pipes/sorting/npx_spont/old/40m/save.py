"""sort marques silico with Kilosort 3.0
takes 60 min (with saving, else 20 min)

  author: steeve.laquitaine@epfl.ch
    date: 06.05.2024
modified: 06.05.2024

usage: 

    sbatch cluster/sorting/marques_silico/40m/save.sbatch

Note:
    - We do no preprocessing as Kilosort3 already preprocess the traces with
    (see code preprocessDataSub()):
    - we set minFR and minfr_goodchannels to 0

% 1) conversion to float32;
% 2) common median subtraction;
% 3) bandpass filtering;
% 4) channel whitening;
% 5) scaling to int16 values

note: to avoid "CUDA_ERROR_ILLEGAL_ADDRESS" we set batch size to default 65792 timepoints

"""

import spikeinterface as si
import spikeinterface.preprocessing as spre
from src.nodes.utils import get_config

# SETUP CONFIG
data_conf, _ = get_config("silico_neuropixels", "concatenated").values()

# SET PATHS
# trace
WIRED_FLOAT32_PATH = data_conf["probe_wiring"]["output_noise_0uV"]
WIRED_INT16_PATH = data_conf["probe_wiring"]["output_noise_0uV_int16"]

# save recording as int16 once
WiredFloat32 = si.load_extractor(WIRED_FLOAT32_PATH)
WiredFloat16 = spre.astype(WiredFloat32, "int16")
WiredFloat16.save(
    folder=WIRED_INT16_PATH,
    format="binary",
    n_jobs=4,
    chunk_memory="40G",
    overwrite=True,
    progress_bar=True
)