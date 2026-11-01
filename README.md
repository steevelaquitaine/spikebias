# Spikebias

[![DOI](https://zenodo.org/badge/885915678.svg)](https://doi.org/10.5281/zenodo.17389967)

author: laquitainesteeve@gmail.com  
contributors: tharayiljoe@gmail.com; milo.imbeni@gmail.com; james.isbister@openbraininstitute.org; michael.reimann@openbraininstitute.org

# System requirements

All software dependencies and operating systems (including version numbers)
Versions: 

* Tested OS:
    * Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K ＠3.2 GHz/5.8 GHz); RTX590 GPU (40GB VRAM)
    * Ubuntu 24.04.1 LTS (16 cores, 62 GB RAM, Intel Xeon Platinum 8259CL ＠2.50GHz)
    * Mac OS X 10.15.7 (8 cores, 8 GB RAM, BuildVersion: 19H2026, Quad-Core Intel Core i7 ＠2.50GHz) 
* Tested Conda versions:
    * 24.11.3
    * 24.9.2
* Tested Software:
  * python 3.9.7
  * python 3.10.8 (demo)
  * the python dependencies are listed in envs/demo.yml for the demo and envs/spikebias.yml for the paper's analyses:
  * Matlab R2022b with: 
    * the Parallel Computing Toolbox (Version 7.7)
    * the Signal Processing Toolbox (Version 9.1) 
    * the Statistics and Machine Learning Toolbox (version 12.4)
  * the spike sorters tested are:
    * kilosort cloned from https://github.com/cortex-lab/KiloSort.git
    * kilosort 2 release: https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.0.2.tar.gz
    * kilosort 2.5 release https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.5.tar.gz
    * kilosort 3 from spikeinterface 0.100.5
    * kilosort 4 from spikeinterface 0.100.5

Versions the software has been tested on: the versions referenced above.

Any required non-standard hardware: 
* most of the raw source code necessary to generate the small intermediate dataset were run on a supercomputer and require MPI and parallel computing on about ten computer nodes.
* spike sorting requires GPU.
* disk storage of about 2TB

# Installation guide

## Instructions

Create and activate the main python virtual environment (spikinterf0_100_5) to run spikebias:

1. move to the root path of the repository and create env with: 

```bash 
conda env create -f envs/spikebias.yml --prefix ./envs/spikebias # create
conda activate ./envs/spikebias # activate
```

2. Run a function from the custom package 

```bash
# move to repository path
import os
os.chdir("/home/jovyan/steevelaquitaine/spikebias/")

# import a module from the custom package
from src.nodes.utils import get_config
from src.nodes.validation import noise

# calculate mean absolute deviation of a recording trace
trace_mad = noise.torch_mad(trace)
```

## Typical install time

Conda install takes 30 minutes.

# Demo

See jupyter notebooks in notebooks/0_demo/

# Instructions for use 

How to run the software on the manuscript's data.

1. You can run the notebooks in notebooks/1_results and notebooks/2_supp_results with the installed virtual environment by running the command below, then selecting the kernel "spikebias" in the notebook:

```bash
python -m ipykernel install --user --name spikebias --display-name "spikebias"
```

# Datasets

* The full sonata circuit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HISHXN **(300 GB)**
    * this is the largest circuit which includes the cortical column cells used for all biophysical simulations
* The O1 sonata circuit: https://zenodo.org/records/11113043 (51 GB)
* The cell id in our experiments are the cell ids in the CELL_ID_SONATA - 1.
* All the cell properties can be retrieved from these circuits.

```bash
dataset/
    00_raw/ (505 GB)
```

# [TODO]

- update the dense_probe1 SortingExtractors uploaded on dandiarchive (dandiset 001250, `sub-003-fitted`). The KS3, KS2.5, KS2, KS and HS 10-minute sortings (`sub-003-fitted_ses-ks3-10m.nwb`, `ses-ks25-10m`, `ses-ks2-10m`, `ses-ks-10m`, `ses-hs-10m`) are duplicates of the dense_probe2 sortings (`sub-004-fitted`, same ses-* files): same number of units and same spike trains. They do not match the dense_probe1 ground truth (`sub-003-fitted_ecephys.nwb`), so all well-sorted ground truth counts are about 0 for depth 1.
    - the correct dense_probe1 sortings are in `dataset/01_intermediate/sorting/dense_probe1/`. Number of units (KS4, KS3, KS2.5, KS2, KS, HS): 94, 238, 518, 512, 693, 29 (dandiarchive duplicates of dense_probe2: 131, 293, 530, 533, 498, 88)
    - the dandiarchive KS4 sorting of dense_probe1 is also a duplicate of dense_probe2 (131 units; the DANDI copy `dataset/01_intermediate/sorting/dense_probe1/SortingKS4_10m_` was removed from the working tree, it is still in git history (commit a6e5958); the correct one has 94 units, in `SortingKS4_10m`). The dandiarchive KS4 sortings of dense_probe2 and dense_probe3 (`SortingKS4_10m_`) are identical to the GitHub ones (all spike trains compared) and were removed from the working tree, they are still in git history (commit a6e5958)