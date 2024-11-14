#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 11.03.2024
## modified: 26.04.2024
##
## usage:
##
##      . cluster/prepro/npx_spont/process_dandihub.sh
##
## duration: 26 min (for 1 hour recording)

## Setup job config

#!/bin/bash -l
#SBATCH -J prep-npx-spont-nwb                                      # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 8 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # setup one GPU
#SBATCH -o ./cluster/logs/cluster/output/prepro/npx_spont_from_nwb/%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/prepro/npx_spont_from_nwb/%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# Requirements:
# 1. create a python spikinterf0_100_5 environment (see envs/spikeinterf...txt)

# setup python3.11 env
source /home/jovyan/steevelaquitaine/assets/envs/spikinterf0_100_5/bin/activate

# add spikebias package to the python path
cd /home/jovyan/steevelaquitaine/spikebias
export PYTHONPATH=$(pwd)

# run pipeline
python3.11 -c "from src.pipes.prepro.npx_spont.process_dandihub import run; run(filtering='butterworth')"