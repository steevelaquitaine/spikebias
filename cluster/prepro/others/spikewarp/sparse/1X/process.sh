#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 17.04.2024
## modified: 17.04.2024
##
## usage:
##
##      sbatch cluster/prepro/others/spikewarp/sparse/1X/process.sh
##
## duration: 2h26 min (for 1 hour recording)

## Setup job config

#!/bin/bash -l
#SBATCH -J prepro-for-spikewarp-sparse-1X                # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # get one GPU
#SBATCH -o ./cluster/logs/cluster/output/spikewarp/prepro/sparse/%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/spikewarp/prepro/sparse/%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com
# load matlab after clean up, compile cuda code and setup python3.9 interpreter via spack env

# load matlab after clean up, compile cuda code and setup python3.9 interpreter via spack env
module purge 
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bs
d/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate

# make package importable
export PYTHONPATH=$(pwd)

# run preprocessing
srun -n 1 python3.9 -c "from src.pipes.prepro.others.spikewarp.sparse.s1X.process import run; run(filtering='butterworth')"