#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 01.09.2024
##
## usage:
##
##      sbatch cluster/model/npx_spont/10m/setup_cebraspike2.sh
##
## duration: 26 min (for 1 hour recording)

## Setup job config

#!/bin/bash -l
#SBATCH -J setup_cebraspike2                             # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH -o ./cluster/logs/cluster/output/slurm_nm_%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/slurm_nm_%x_id_%A.err   # set log error file path
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

# 2. Setup cebraspike virtual environment:
python3.9 -m venv /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/cebraspike2 # create env
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/cebraspike2/bin/activate
/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/cebraspike/bin/python3.9 -m pip install -r /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/envs/cebraspike.txt
