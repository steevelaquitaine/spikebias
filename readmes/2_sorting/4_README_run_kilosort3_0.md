# Blue Brain Project spike sorting scripts

authors: steeve.laquitaine@epfl.ch  
modified from milo.imbeni@epfl.ch, michael.reimann@epfl.ch  
with feedbacks from: joseph.tharayil@epfl.ch  

Built to compare real firing rates with firing rates detected from MEAs inserted into anaesthetised rat somatosensory cortex.

Needs additional dataset and output folder to store raw data and processed recordings.

[TODO]:
- test whether to load all modules and all dependencies including bluepy at once at the beginning for the entire project.

# Table of Contents

- [Paths structure](#directory) .      
- [Setup the project](#setup-the-project) .   
- [In-silico simulation](#in-silico-simulation) .   
  - [Edit channel weights](#edit-channel-weights)
  - [Launch campaigns](#launch-campaign)
  - [Setup processing env ](#setup-simulation-environment) .   
  - [Data engineering](#data-engineering) .   
  - [Preprocessing](#preprocessing) .    
  - [Get ground truth spikes](#get-ground-truth-spikes)
- [Sorting](#sort-firing-rates) .   
  - [Reyes dataset](#reyes-dataset) .   
    - [Setup dataset environment](#setup-dataset-environment) .   
    - [Dataset preprocessing](#dataset-preprocessing) .     
    - [Sorter](#sorters) .       
  - [Buccino dataset](#buccino-paper-dataset) .   
- [Deepnet for sorting](#deepnet) 
- [Run notebooks][#notebooks] 
- [Useful info](#useful-info) .   
  - [Setup SSH authentication to Gitlab](#setup-ssh-authentication-to-gitlab)
  - [Connect to the login node](#connect-to-the-login-node)
  - [Run an interactive job on the cluster](#run-an-interactive-job-on-the-cluster)
  - [Setup a virtual environment](#setup-a-virtual-environment)
  - [Port forward compute node to local](#port-forward-compute-node-to-local)
  - [Open a notebook on a compute node in a Spack environment](#open-a-notebook-on-a-compute-node-in-a-spack-environment)
  - [Open VScode notebook on a compute node](#open-vscode-notebook-on-a-compute-node)
  - [Bluepy simulator](#bluepy-simulator) .   
  - [Free ports](#free-ports) .   
  - [BBP workflow](#bbp-worlflow) .   
  - [References](#references) .   

# Paths structure

Project code directory: 

```bash
git clone git@bbpgitlab.epfl.ch:conn/personal/imbeni/spike-sorting.git
```

Project data directory:

```
- /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/
  - reyes4s/
    - 7/
      - BlueConfig                                    # input
  - sorting/
    - output/
      - 0_silico/
        - Hex0_4s_reyes128/
          - Hex0_4s_reyes128_real_wfs/                # input
          - results/                                  # input
            - spikes                                  # output
          - Hex0_4s_reyes128_rec/                     # output            
          - Hex0_4s_reyes128_traces.pkl               # output
          - Hex0_4s_reyes128_spiketrains.pkl          # output
          - Hex0_4s_reyes128_true_spikes/             # output
- /gpfs/bbp.cscs.ch/project/proj68/scratch/tharayil/
  - coeffsreyespuerta.h5                              # input
```

Working on new standardized project data directory:
TODO: 
- move reyes4s folder into a raw/ folder
- prefix with a step order: 0_raw, 1_dataeng, 2_sorting 
- Keep only campaign level files to save space 
  - delete all chunks and simulation-level files after stacking

This is the final folder path:

```
- /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/
  - dataeng/
    - 0_silico/
      - Hex0_4s_reyes128/
        - campaign/
          - raw/    
            - cells/
            - spiketrains.pkl
            - traces.pkl
          - preprocessed/
            - traces.pkl
        - channel_locations/
        - Hex0_4s_reyes128_locations.npy
        - simulations/
          - 1/
            - chunks/ 
              - cells/
              - spikes0.pkl
              - ...
              - spikes29.pkl
              - traces0.pkl
              - ...
              - traces29.pkl              
            - stacked/
          - 2/
          - ...
          - 10/
  - reyes4s/
    - 1/
      - BlueConfig                                    # input
      - ...
    - 2/
      - BlueConfig                                    # input
      - ...
    - ...
    - 10/
      - BlueConfig                                    # input
      - ...
  - sorting/
    - output/
      - 0_silico/
        - Hex0_4s_reyes128/
          - Hex0_4s_reyes128_real_wfs/                # input
          - results/                                  # input
            - spikes                                  # output
          - Hex0_4s_reyes128_rec/                     # output            
          - Hex0_4s_reyes128_traces.pkl               # output
          - Hex0_4s_reyes128_spiketrains.pkl          # output
          - Hex0_4s_reyes128_true_spikes/             # output
    
  
- /gpfs/bbp.cscs.ch/project/proj68/scratch/tharayil/
  - coeffsreyespuerta.h5                              # input
```


# Setup the project

Clone the project's codebase (you do it once). My project path `proj_path` on the cluster's login machine
is `/gpfs/bbp.cscs.ch/project/proj68/home/laquitaine/spike-sorting/'

```bash
# connect to server
ssh laquitai@bbpv1.epfl.ch  

# create home directory
cd /gpfs/bbp.cscs.ch/project/proj68/home/ 
mkdir laquitai
cd laquitai/

# clone project and create feature branch
git clone https://bbpgitlab.epfl.ch/conn/personal/imbeni/spike-sorting.git
git branch steeve-cleanup
```

# In-silico simulation

## Create/edit channel weights

```bash
# Create python virtual environment
python3.9 -m venv env_silico
source env_silico/bin/activate

# Then pip3.9 install
pip3.9 install -r requirements_silico.txt

# run channel weight editing pipeline
python -m src.pipes.silico.channel_weights
```

## Launch campaign

1. Install bbp_workflow: see [README for bbp workflow](readmes/bbp_workflow.md)

Your simulation data will be saved here `/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw`
under the template name <probe>_lfp_<duration>_<date>. For example: `neuropixels_lfp_10m_2023_02_19`.

## Setup simulation environment

1. Run in the cluster's interactive mode: from the login machine, allocate resource to a compute node, e.g., r2i3n3 (different from the login node): 

```bash
ssh laquitai@bbpv1.epfl.ch
salloc --nodes=1 -t 06:00:00 --account=proj68 --partition=interactive --constraint=volta --gres=gpu:1 --mem=0
# You should see: 
# salloc: Granted job allocation 1022328
# salloc: Waiting for resource configuration
# salloc: Nodes r2i3n3 are ready for job
# Hostname: r2i3n3
# User: laquitai
```

2. (Necessary) Clean-up before you start:

    1. Remove all previously installed packages by pip in ~/.local/

    ```bash
    rm -rf ~/.local/
    ```

    2. Ensure that no module is loaded on the cluster:

    ```bash
    module purge
    ```

2. Create and activate the Spack environment for preprocessing:

```bash
module load spack
spack --version
# 0.17.1
spack env create spack_env spack_silico_prepro.yaml                                       # create spack env.
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh  # setup for activation
```

* Troubleshooting:  
  * if Spack_env already exists: run ```spack env remove spack_env```

3. Activate and install Spack packages: 

```bash
spack env activate spack_env -p
spack install                                       # installs root specs in spack.yaml and its dependencies
spack load python@3.9.7                             # load installed external dependency python 3.9.7
spack find --loaded
# you should see python@3.9.7 loaded packages (not 0 loaded packages)
which python3.9
# /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/stage_externals/install_gcc-11.2.0-skylake/python-3.9.7-yj5alh/bin/python3.9
```
note: you load the external Python3.9 to enable its use. It permits to install pip3.9 and proper project package versions (e.g., pandas 1.5.2).  

4. Install python packages with pip3.9:

```bash
python3.9 -m ensurepip --default-pip
pip3.9 --version
# pip 21.2.3 from /gpfs/bbp.cscs.ch/home/laquitai/.local/lib/python3.9/site-packages/pip (python 3.9)
pip3.9 install --user -r requirements_silico.txt
```

* Troubleshooting:  
  * If these commands fail to reinstall pip for python3.9:    
    * action: remove .local/. 
    * explanation: pip3.9 installs packages in ~/.local/lib. Removing .local/ will enable package reinstall.

4. Install `bluepy` and `bluepy-config`: copy Joseph Tharayil's bluepy-config repo to your home (where you have pip install pernission):

```bash
cp -r /gpfs/bbp.cscs.ch/home/tharayil/bluepy-configfile ~/  # copy bluepy-config
pip3.9 install --user ~/bluepy-configfile/                  # install
```

Install bluepy. Move to your home directory e.g., /gpfs/bbp.cscs.ch/home/laquitai. You need to move there because you have pip install permission. You will be denied install permissions in other paths. Then clone bluepy and checkout branch lfp-reports.
Make sure your ssh private key has been setup before for authentication to Gitlab.

```bash
git clone git@bbpgitlab.epfl.ch:nse/bluepy.git ~/ . # clone in home path
cd ~/bluepy/
git checkout lfp-reports                            # checkout the lfp package (commit 6f4ee4aa9798c6083659c333ca2277fa3871e521)
pip3.9 install --user .                             # install
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/ # move back to project path
```

5. Troubleshooting

* if Python is not in spack path, there might be too matching packages `spack load python@3.9.7/desired_hash` (when you run spack load)

## Data engineering

In brief: 

1. Chunk each simulation of a campaign of N long simulations into C chunks, extract and save traces and spikes (`chunking.py`)
2. Stack and save stacked trace and spike files (`stacking.py`)
3. Load for analysis 

Steps: 

1. Chunk all simulations. This steps can chunks simulations read, their traces and spikes and write them in `.pkl` files, one file for each chunk.
The `chunking_parallelized` module parallelizes the pipeline and running it on the cluster.

```bash
# move to project root
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/  
```

```bash
# activate your spack environment then run e.g., 
python3 src/pipelines/simulation/dataeng/chunking.py --conf 2023_01_13
```

For cluster processing, edit `parallelize_chunking.sbatch`, set the experiment (e.g.,`silico_reyes`), stored in `conf/` and the run date `2023_01_13`.
The number of nodes must be set to the number of simulations in the campaign (e.g., here `-n 10`).

```bash
srun -n 10 python3 /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/src/nodes/dataeng/silico/chunking_parallelized.py --exp supp/silico_reyes --conf 2023_01_13
```

Then run:

```bash
sbatch sbatch/parallelize_chunking.sbatch
```

2. Stack all chunks across simulations into a campaign file: 

```bash
python3 src/pipelines/simulation/dataeng/stacking.py --conf 2023_01_13
```

## Preprocessing 

### CLI

Activate spack environment then run:

```bash
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/ # move back to project path
python3.9 app.py simulation --exp supp/silico_reyes --pipeline preprocess --conf 2023_01_13
```

### Notebook

TODO:
- update t. Possible in VSCODE:

1. Forward a compute node's port to a port on your local machine (see [Port forward compute node to local](#port-forward-compute-node-to-local))
2. Activate your spack env from the same terminal:

```bash
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
spack env activate spack_env -p
```

3. Run jupyter notebook from the same terminal:

```bash
jupyter notebook --port 8888
```

## Get ground truth spikes

```bash
python3.9 app.py simulation --pipeline sort ground_truth
```

## Sort spikes



### Buccino paper dataset

See instructions in [Buccino notebook](https://spikeinterface.github.io/blog/ground-truth-comparison-and-ensemble-sorting-of-a-synthetic-neuropixels-recording/)

1. Create virtual environment:
  2. Create spack env:

  ```bash
  cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
  module load spack
  . /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
  spack env activate spack_env -p
  spack load python@3.9.7
  ```

  3. Create buccino env:

  ```bash
  python3.9 -m venv env/buccino_env
  source env/buccino_env/bin/activate
  ```

2. Install dependencies:

```bash
pip3.9 install -r requirements_buccino.txt
```

3. Download `sub-MEAREC-250neuron-Neuropixels_ecephys.nwb` file (28 GB):

```bash
dandi download https://api.dandiarchive.org/api/assets/6d94dcf4-0b38-4323-8250-04fdc7039a66/download/
```

# Sort firing rates

## Reyes dataset

### Setup dataset environment

1. Run in the cluster's interactive mode: from the login machine, allocate resource to a compute node, e.g., r2i3n3 (different from the login node): 

```bash
ssh laquitai@bbpv1.epfl.ch
salloc --nodes=1 -t 06:00:00 --account=proj68 --partition=interactive --constraint=volta --gres=gpu:1 --mem=0
# You should see: 
# salloc: Granted job allocation 1022328
# salloc: Waiting for resource configuration
# salloc: Nodes r2i3n3 are ready for job
# Hostname: r2i3n3
# User: laquitai
```

2. (Necessary) Clean-up before you start:

```bash
rm -rf ~/.local/  # remove previously installed pip packages in ~/.local/ 
module purge      # ensure that no module is loaded on the cluster:
```

2. Create and activate the Spack environment for preprocessing:

```bash
module load spack
spack --version
# 0.17.1
spack env create spack_env spack_reyes_prepro.yaml                          # create spack env.
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh  # setup for activation
```

* Troubleshooting:  
  * if Spack_env already exists: run ```spack env remove spack_env```

3. Activate and install Spack packages: 

```bash
spack env activate spack_env -p
spack install                                       # installs root specs in spack.yaml and its dependencies
spack load python@3.9.7                             # load installed external dependency python 3.9.7
spack find --loaded
# you should see python@3.9.7 loaded packages (not 0 loaded packages)
which python3.9
# /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/stage_externals/install_gcc-11.2.0-skylake/python-3.9.7-yj5alh/bin/python3.9
```

note: you load the external Python3.9 to enable its use. It permits to install pip3.9 and proper project package versions (e.g., pandas 1.5.2).  

4. Install python packages with pip3.9:

```bash
python3.9 -m ensurepip --default-pip
pip3.9 --version
# pip 21.2.3 from /gpfs/bbp.cscs.ch/home/laquitai/.local/lib/python3.9/site-packages/pip (python 3.9)
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/  # move to project path
pip3.9 install --user -r requirements_silico.txt                  # install project packages
```

### Dataset preprocessing 

```bash
python3.9 reyes_preprocess_spont.py
# It takes 15 min for 10 min of recording (1.5 min / min of recording) and 
# saves a 5.8 GB dataset

# > should see:
# salloc: Granted job allocation 1022080
# salloc: Waiting for resource configuration
# salloc: Nodes r1i7n14 are ready for job
# Reading raw
# Fixing scaling
# Load and clean recording
# Preprocessing and saving
# write_binary_recording with n_jobs = 1 and chunk_size = None
# salloc: Relinquishing job allocation 1022080
```

### Sorters

1. Clone `kilosort2, ``kilosort3`, `hdsort`, `ironclust`in <proj_path>/sorters_packages/:

```bash
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/sorters_packages/ 
git clone https://github.com/MouseLand/Kilosort.git
git clone https://github.com/MouseLand/Kilosort2          # kilosort2
git clone https://git.bsse.ethz.ch/hima_public/HDsort.git
git clone https://github.com/jamesjun/ironclust
```

2. Setup matlab and GPU: 

```bash
# load module dependencies
# module load unstable
module load matlab                # version is R2019b Update 2 (9.7.0.1247435), for kilosort3
# module load hpe-mpi/2.25.hmpt   # for Spyking-circus 

# setup GPU
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/sorters_packages/Kilosort/CUDA
matlab -batch mexGPUall
# > you should see:
# MATLAB is selecting SOFTWARE OPENGL rendering.
# Building with 'nvcc'.
# ...
# MEX completed successfully.
```

3. Run interactively (e.g., it can be fast for one `kilosort3`): 

```bash
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/ 
python3.9 reyes_128_spont_fullsort_kilo.py
```

... or submit a batch job on a cluster node with GPU:

```bash
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/  # back to project root

# run sorting and check stdout .out file in cluster/output/
sbatch sorting.sbatch
# > should see:
# sbatch: INFO: Activating auto partition selection plugin, please report errors to HPC/CS
# Submitted batch job 1021835
```

4. Test other sorters (`herdingspikes`, `hdsort`):

```bash
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/ 
python3.9 reyes_128_spont_fullsort_allsorters.py
```

note:
* `SpykingCircus` takes a while (even for a 5 min secs recording).
* `herdingspikes` takes less than 5 min (for 5 min secs recordings).

VALIDATED UNTIL THERE !!

### Spike analysis

1. From the login machine, allocate resource on a compute node, e.g., r2i3n0 (different from the login node): 

```bash
ssh laquitai@bbpv1.epfl.ch
salloc --nodes=1 -t 03:00:00 --account=proj68 --partition=interactive --constraint=volta --gres=gpu:1 --mem=0
# You should see: 
# salloc: Granted job allocation 1022328
# salloc: Waiting for resource configuration
# salloc: Nodes r1i7n6 are ready for job
# Hostname: r2i3n0
# User: laquitai
```

2. Enter in Vscode Palette: `remote-ssh connect to host...` then enter `ssh -J bbpv1.epfl.ch laquitai@r2i3n0`.
This will create the .ssh/config file below.

```config
Host r2i3n3
  HostName r2i3n3
  ProxyJump bbpv1.epfl.ch
```
That's it. Click on a notebook. 

Before running notebooks you need to purge the modules and reinstall the dependencies 
in your virtual environment. This solves dependency conflicts between loaded modules 
libraries and jupyter notebook dependencies (e.g., with numpy). You will need to reload
the modules to re-run the preprocessing and sorting pipelines.

```bash
module purge
pip install --user -r requirements.txt
```

Then run the notebook. 

see (4).


# Deepnet

# Run notebooks

see readmes/All_figures.md

# Useful info

## Setup SSH authentication to Gitlab

1. Check for an existing ssh key:

```bash
ls -al ~/.ssh
# > should list things like that:
# authorized_keys
# id_rsa
# id_rsa.pub
# known_hosts
```

2. Add your existing ssh public key to your Gitlab:

3. Show and copy your public key if it exists

```bash
cat ~/.ssh/id_rsa.pub
# > should display your key
```

4. Then open Gitlab - click on your user icon (top right) - preferences - SSH keys tab (left) - paste in box - add key

## Connect to the login node

1. (Do once the first) Check that ssh authentication is available on the remote server and locally:

```bash
ssh
# < should list ssh arguments
```

1. Install remote development installation pack in vscode locally:  
  1. Click on extension icon
  2. Enter and select "remote development"
  3. Install

2. Check that you can connect to the remote server:

```bash
ssh laquitai@bbpv1.epfl.ch   
```

3. Configure vscode for remote connection: 

4. Open vscode palette (ctrl+shift+p), enter and select "Remote-SSH: Connect to Host ..." 
(this is enabled by the remote development extension that installed previously)

5. Configure SSH host and enter your <user>@<host> (example: "laquitai@bbpv1.epfl.ch") and password

A new vscode GUI opens, connected to the remote server.

Done.

## Run an interactive job on the cluster

To run an example job in interactive mode on a single node, run in the terminal of the login server:

```bash
salloc --nodes=1 --account=proj68 --partition=interactive --constraint=cpu --mem=0 srun -n 1 python3 Tutorials/test_interactive_node.py
```

note: do not run with sbatch.

## Setup a virtual environnent

* see module import: https://bbpteam.epfl.ch/project/spaces/display/BSD/Python+Usage+on+BB5
* see Spack: https://drive.google.com/file/d/1923icj7PNP7p-CUiSShVJsJ_xSvvXmmK/view
* see a getting started: https://bbpteam.epfl.ch/project/spaces/display/BBPHPC/A+Simple+getting+started+guide+for+working+with+CoreNeuron+and+Neurodamus

## Port forward compute node to local

(The only method I could make work so far with a Spack environment).

1. From the login machine, allocate resource on a compute node, e.g., r2i3n0 (different from the login node): 

```bash
ssh laquitai@bbpv1.epfl.ch
salloc --nodes=1 -t 06:00:00 --account=proj68 --partition=interactive --constraint=volta --gres=gpu:1 --mem=0
# You should see: 
# salloc: Granted job allocation 1022328
# salloc: Waiting for resource configuration
# salloc: Nodes r1i7n6 are ready for job
# Hostname: r2i3n0
# User: laquitai
```

2. Open a new terminal on your local machine (do not close the previous terminal as it would end your compute node allocation), port forward to an available port of the login node (e.g., 5678). Be careful of any warning that the port is unavailable (see under Troubleshooting sectiob how to free a port). From the login node, port forward to the noted compute node r2i3n0's port (e.g., 8888),
ssh to compute node, run jupyter notebook. Be careful that associated port is 8888 as set above.

```bash
ssh -L 8888:localhost:5678 laquitai@bbpv1.epfl.ch # port forward the login node port to the compute node port
ssh -L 5678:localhost:8888 r2i3n0 # ssh to compute node
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting # move to project path
```

## Open a notebook on a compute node

(The only method I could make work so far with a Spack environment).

1. From the login machine, allocate resource on a compute node, e.g., r2i3n0 (different from the login node): 

```bash
ssh laquitai@bbpv1.epfl.ch
salloc --nodes=1 -t 06:00:00 --account=proj68 --partition=interactive --constraint=volta --gres=gpu:1 --mem=0
# You should see: 
# salloc: Granted job allocation 1022328
# salloc: Waiting for resource configuration
# salloc: Nodes r1i7n6 are ready for job
# Hostname: r2i3n0
# User: laquitai
```

2. Open a new terminal (not in VSCODE, your jupyter notebook's pip will break) on your local machine (do not close the previous terminal as it would end your compute node allocation), port forward to an available port of the login node (e.g., 5678). Be careful of any warning that the port is unavailable (see under Troubleshooting sectiob how to free a port). From the login node, port forward to the noted compute node r2i3n0's port (e.g., 8888),
ssh to compute node, run jupyter notebook. Be careful that associated port is 8888 as set above.

```bash
ssh -L 8888:localhost:5678 laquitai@bbpv1.epfl.ch # port forward the login node port to the compute node port
ssh -L 5678:localhost:8888 r2i3n0 # ssh to compute node
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting # move to project path
```

3. Setup your Spack environment (see above) with installed packages) and run notebook: 

```bash
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh # add spack to path
spack env activate spack_env -p # your spack_env must have been created as described above
pip3.9 install jupyter notebook # install jupyter and notebook packages
jupyter notebook --port 8888 # call jupyter
```

4. Open the notebook url printed, and keep your local machine's terminal opened.

5. Troubleshooting

* Make sure that the ports are available. If not see "Free ports" section.

## Open VScode notebook on a compute node 

1. Allocate resource to a compute node via the login node: 

```bash
ssh laquitai@bbpv1.epfl.ch  # connect to the login node
salloc --nodes=1 -t 06:00:00 --account=proj68 --partition=interactive --constraint=volta --gres=gpu:1 --mem=0 # allocate a compute node
```

2. Add the compute node as a new Host:   
  1. Open command palette (Ctrl + Shift + P)
  2. Enter "Remote SSH: Connect to Host ..."
  3. Click "+ Add New SSH Host"
  4. Enter "ssh -J bbpv1.epfl.ch laquitai@r2i3n3"
3. Open command palette (Ctrl + Shift + P)
  1. "Select an interpreter": 
  2. "+ enter path path ...": enter python 3.9.7's path (printed by `which python` call in terminal)
  3. "Select an interpreter": select python 3.9.7  
4. Open a notebook from VSCODE's explorer.
5. Troubleshooting:  
  * You get an error when running jupyter notebook on an allocated cluster node
      * solution: delete .vscode-server from your home on the cluster's login machine.

## Bluepy simulator

* read doc: https://bbpteam.epfl.ch/documentation/projects/bluepy/latest/

## Free ports

* free port 8888 by killing the process using it.

```bash
lsof -i:8888
kill $(lsof -t -i:8888)
lsof -i:8888
```

* Conflict: packages loaded with `module load ...` overwrite packages installed with pip. 

## bbp workflow

see [bbp worflow readme](readmes/bbp_workflow.md)

## References

(1) https://bbpteam.epfl.ch/project/spaces/display/SDKB/HPC+Service#HPCService-BB5Tutorial
(2) https://bbpteam.epfl.ch/project/spaces/display/INFRA/Slurm+cheat+sheet
(3) https://bbpteam.epfl.ch/project/spaces/display/SDKB/HPC+Service
(4) https://bbpteam.epfl.ch/project/spaces/pages/viewpage.action?pageId=22351713
(5) https://spack-tutorial.readthedocs.io/en/latest/tutorial_basics.html 
(6) https://bbpteam.epfl.ch/project/spaces/display/BBPHPC/How-to+articles
  