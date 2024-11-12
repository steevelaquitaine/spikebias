#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 11.03.2024
## modified: 26.04.2024
##
## usage:
##
##      sbatch cluster/processing/marques_silico/process_40m_noise_none_gain_1e6.sbatch
##
## duration: 26 min (for 1 hour recording)

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
#SBATCH -o ./cluster/logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# load matlab after clean up, compile cuda code and setup python3.9 interpreter via spack env
module purge 
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate

# add custom package to python path
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
export PYTHONPATH=$(pwd)

# run pipeline
srun -n 1 python3.9 -c "from src.pipes.prepro.marques_silico.process_40m_noise_none_gain_1e6 import run; run(filtering='butterworth')"