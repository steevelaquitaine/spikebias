#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 13.12.2023
##  updated: 13.12.2023
##
## usage:
##
##      sbatch cluster/metadata/label_marques_silico_2023_10_18.sbatch
##
## stats: 2h20 for 3 concatenated campaigns

## Setup job config

#!/bin/bash -l
#SBATCH --nodes 1                                             # Use 1 node
#SBATCH -t 00:30:00                                      # Set 30 min time limit
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

# setup python3.9 interpreter via spack env
module purge 
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/envs/npx_10m_384ch_unit_classes/bin/activate

# add my custom package to python path
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
export PYTHONPATH=$(pwd)

# run
srun python3.9 -c "from src.pipes.metadata.silico_npx_2023_10_18 import label_layers; label_layers()"