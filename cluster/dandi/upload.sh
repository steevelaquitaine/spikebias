#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
## usage:
##
##      sbatch cluster/dandi/upload.sh
##
## stats: 21 mins

## Setup job config

#!/bin/bash -l
#SBATCH -J dandi-upload                                 # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # get one GPU
#SBATCH -o ./cluster/logs/cluster/output/dandi/%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/dandi/%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# setup
module purge 
module load archive/2023-03
module load python/3.10.8
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/   
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/dandi2/bin/activate

# run pipeline
srun -n 1 python -m src.pipes.dandi_upload.dandi_upload