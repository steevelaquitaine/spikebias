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
#SBATCH -t 01:00:00                                 # Set 1 hour time limit
#SBATCH -p prod                                     # Submit to the production 'partition'
#SBATCH -C "cpu"                                    # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                 # to allocate whole node
#SBATCH --account=proj68                            # your project number
#SBATCH --mem=0                                     # allocate entire memory to the job
#SBATCH --constraint=volta                          # setup node with GPU
#SBATCH --gres=gpu:1                                # setup one GPU
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path

module load matlab
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/sorters_packages/Kilosort/CUDA
matlab -batch mexGPUall
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
spack env activate spack_env -p
spack load python@3.9.7
srun -n 1 python3.9 /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/src/pipes/sorting/run_kilosort3.py --conf 2023_01_13