#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 20.07.2023
## modified: 07.12.2023
##
## usage:
##
##      sbatch cluster/full_processing/silico_horvath_probe2.sbatch
##
## stats: 2h20 for 3 concatenated campaigns

## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 8 hour time limit
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

# load matlab after clean up, compile cuda code and setup python3.9 interpreter via spack env
module purge 
module load matlab/r2019b
cd /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/sorters/Kilosort3/CUDA
matlab -batch mexGPUall
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/envs/npx_10m_384ch_unit_classes/bin/activate

# add custom package to python path
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
export PYTHONPATH=$(pwd)

# concatenate and preprocess traces and spikes
srun -n 1 python3.9 -m src.pipes.prepro.process_silico_concat_horvath_probe2

# sort traces
srun -n 1 python3.9 -m src.pipes.sorting.sort_concat_silico_horvath_2

# compute and save metadata
srun -n 1 python3.9 /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/src/pipes/postpro/postpro_silico_horvath.py silico_horvath concatenated/probe_2