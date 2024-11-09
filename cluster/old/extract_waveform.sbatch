#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
## Usage:
##
## In python: 
##      os.system('sbatch sbatch/sort_with_kilosort3.sbatch 2023_01_13')
##
## In the terminal:
##      edit $conf_date with a simulation conf: e.g., 2023_01_13 then run: 
##      sbatch sbatch/sort_with_kilosort3.sbatch

## Setup job config


#!/bin/bash -l
#SBATCH -N 1                                        # Use 1 node
#SBATCH -t 03:00:00                                 # Set 3 hour time limit
#SBATCH -p prod                                     # Submit to the production 'partition'
#SBATCH -C "cpu"                                    # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                 # to allocate whole node
#SBATCH --account=proj83                            # your project number
#SBATCH --mem=0                                     # allocate entire memory to the job
#SBATCH --constraint=volta                          # setup node with GPU
#SBATCH --gres=gpu:1                                # setup one GPU
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path

cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
source env_kilosort_silico/bin/activate
# srun -n 1 python3.9 /gpfs/bbp.cscs.ch/data/project/proj68/home/laquitai/spike-sorting/src/pipes/postpro/waveform.py
srun -n 1 python3.9 -m src.pipes.postpro.waveform