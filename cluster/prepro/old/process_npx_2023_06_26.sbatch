#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 19.10.2023
## modified: 19.10.2023
##
## usage:
##
##      sbatch cluster/processing/process_npx_2023_06_26.sbatch
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
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# load matlab after clean up, compile cuda code and setup python3.9 interpreter via spack env
module purge 
module load spack
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/sfn_2023/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/envs/npx_10m_384ch_unit_classes/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.prepro.npx_2023_06_26