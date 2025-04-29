# Blue Brain Project spike sorting scripts

author: laquitainesteeve@gmail.com

# System requirements

All software dependencies and operating systems (including version numbers)
Versions: 

* Tested OS for demo:
    * Ubuntu 24.04.1 LTS (16 cores, 62 GB RAM, Intel Xeon Platinum 8259CL ＠2.50GHz)
    * Mac OS X 10.15.7 (8 cores, 8 GB RAM, BuildVersion: 19H2026, Quad-Core Intel Core i7 ＠2.50GHz) 
* Tested Conda versions:
    * 24.11.3
    * 24.9.2
* Software dependencies:
  * python 3.9.7
  * python 3.10.8 (demo)
  * the python dependencies are listed in envs/demo.yml for the demo and envs/spikebias.yml for the paper's analyses:
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
