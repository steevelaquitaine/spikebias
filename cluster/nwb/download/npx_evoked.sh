#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 17.04.2024
## modified: 17.04.2024
##
## usage:
##
##      sh cluster/nwb/download/npx_evoked.sh
##
## stats: 20 mins

## Setup job config

#!/bin/bash -l
#SBATCH -J nwb-download-npx_evoked                       # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # get one GPU
#SBATCH -o ./cluster/logs/cluster/output/nwb/download/%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/nwb/download/%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/dandi2/bin/activate

# run pipeline
srun -n 1 python -m src.pipes.nwb.download.npx_evoked