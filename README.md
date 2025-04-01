# Blue Brain Project spike sorting scripts

author: laquitainesteeve@gmail.com

# System requirements

All software dependencies and operating systems (including version numbers)
Versions: 

* OS: Ubuntu 24.04.1 LTS and Mac OS X 10.15.7 (BuildVersion: 19H2026)  
* Software dependencies: python 3.11.10 and the dependencies listed in envs/  

Versions the software has been tested on: the versions referenced above.

Any required non-standard hardware: 
* part of the raw source code necessary to generate the small intermediate dataset requires MPI and parallel computing on at least ten computer nodes.
* spike sorting require GPU.

# Installation guide

## Instructions

Create and activate the main python virtual environment (spikinterf0_100_5) to run spikebias:

```bash
python3.11 -m venv /home/jovyan/steevelaquitaine/spikebias/envs/spikinterf0_100_5 # create env
source /home/jovyan/steevelaquitaine/spikebias/envs/spikinterf0_100_5/bin/activate
python3.11 -m pip install -r /home/jovyan/steevelaquitaine/spikebias/envs/spikinterf0_100_5_dandihub.txt
```

## Typical install time




