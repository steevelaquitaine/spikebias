#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 23.11.2023
## modified: 31.01.2024
##
## usage:
##
##      sbatch cluster/prepro/dense_spont/process_probe3.sh
##
## duration: 2:30 hours (for 40 min recording)

## Setup job config

#!/bin/bash -l
#SBATCH -J prep-dense-spont3                              # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 8 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # setup one GPU
#SBATCH -o ./cluster/logs/cluster/output/prepro/dense_spont/%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/prepro/dense_spont/%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# setup python3.9 env via spack env
module purge 
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate

# add custom package to python path
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/   
export PYTHONPATH=$(pwd)

# run pipeline
srun -n 1 python3.9 -c "from src.pipes.prepro.dense_spont.process import run; run(experiment='dense_spont', run='probe_3', noise_tuning=2)"