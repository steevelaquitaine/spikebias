#!/bin/bash -l

## author: steeve.laquitaine@epfl.ch
##
## modified: 01.07.2023
##
## usage:
##
##      sbatch cluster/sort_janelia.sbatch
##
## stats: 233 secs

## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 01:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # setup one GPU
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# clean up
module purge 
module load matlab/r2019b

# compile cuda code
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/sorters/Kilosort3/CUDA
matlab -batch mexGPUall

# access python3.9 interpreter via spack env
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate spack_env_ks3_janelia -p

# activate python env
source envs/env_spikeforest/bin/activate
 
# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.sort_hybrid_janelia_with_KS3_test
