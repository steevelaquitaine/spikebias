#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 17.04.2024
## modified: 17.04.2024
##
## usage:
##
##      sbatch cluster/sorting/npx_evoked/40m/ks4.sh
##
## stats: 21 mins

## Setup job config (32 GB GPUs (2x more than before))

#!/bin/bash -l
#SBATCH -J ks4-evoked-40m                               # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 8 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=gpu_32g                             # setup node with GPU with 32 GB RAM (2x more than before)
#SBATCH -o ./cluster/logs/cluster/output/sorting/npx_evoked/40m/slurm_nm_%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/sorting/npx_evoked/40m/slurm_nm_%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

module purge 
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.npx_evoked.40m.ks4