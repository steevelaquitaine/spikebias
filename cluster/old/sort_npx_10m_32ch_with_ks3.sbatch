#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
## modified: 01.07.2023
##
## usage:
##
##      sbatch sbatch/sort_npx_10m_32ch_with_ks3.sbatch
##
## stats: 2h

## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 10:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # setup one GPU
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path

# clean up
module purge 
module load matlab/r2019b

# compile cuda code
cd /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/cns_2023/sorters/Kilosort3/CUDA
matlab -batch mexGPUall

# access python3.9 interpreter via spack env
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/cns_2023_on_git/
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate spack_python3_9 -p

# activate python env
source /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/cns_2023/envs/sort_npx_10m_32ch_with_ks3/bin/activate

# run pipeline
srun -n 1 /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/cns_2023/envs/sort_npx_10m_32ch_with_ks3/bin/python3.9 -m src.pipes.sorting.sort_npx_10m_32ch_with_ks3