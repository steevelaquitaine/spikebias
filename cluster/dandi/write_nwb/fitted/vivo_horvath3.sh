#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 17.04.2024
## modified: 17.04.2024
##
## usage:
##
##      sbatch cluster/dandi/write_nwb/fitted/vivo_horvath3.sh
##
## stats: 1h30

## Setup job config

#!/bin/bash -l
#SBATCH -J dandi-fitted-vivo_horvath3                        # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 24:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # get one GPU
#SBATCH -o ./cluster/logs/cluster/output/dandi/write_nwb/vivo_horvath3/%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/dandi/write_nwb/vivo_horvath3/%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

module purge 
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/envs/spikinterf0_100_5/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.dandi.write_nwb.fitted.vivo_horvath3