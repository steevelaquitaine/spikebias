# Blue Brain Project spike sorting scripts

author: laquitainesteeve@gmail.com

# System requirements

All software dependencies and operating systems (including version numbers)
Versions: 

* OS: Ubuntu 24.04.1 LTS and Mac OS X 10.15.7 (BuildVersion: 19H2026) 
* Conda 24.11.3 and 24.9.2
* Software dependencies:
  * python 3.11.10
  * dependencies listed in envs/spikebias.yml for the main analyses:
  * spike sorters:
    * kilosort cloned from https://github.com/cortex-lab/KiloSort.git
    * kilosort 2 release: https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.0.2.tar.gz
    * kilosort 2.5 release https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.5.tar.gz
    * kilosort 3 from spikeinterface 0.100.5
    * kilosort 4 from spikeinterface 0.100.5

Versions the software has been tested on: the versions referenced above.

Any required non-standard hardware: 
* part of the raw source code necessary to generate the small intermediate dataset requires MPI and parallel computing on at least ten computer nodes.
* spike sorting require GPU.

# Installation guide

## Instructions

Create and activate the main python virtual environment (spikinterf0_100_5) to run spikebias:

1. move to the root path of the repository and create env with: 

```bash 
conda env create -f envs/spikebias.yml --prefix ./envs/spikebias # create
conda activate ./envs/spikebias # activate
```

## Typical install time

Conda install takes 30 minutes.

# Demo [TODO]


# Instructions for use 

How to run the software on the manuscript's data.

1. You can run the notebooks with the virtual environment by running the command below, then selecting the kernel in the notebook

```bash
python -m ipykernel install --user --name spikebias --display-name "spikebias"
```
