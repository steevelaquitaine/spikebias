#!/bin/bash -l

##   author: steeve.laquitaine@epfl.ch
##     date: 17.01.2023
## modified: 01.02.2024
##
##    Usage:
##
##      sbatch cluster/postpro/horvath_vivo/save_metadata_probe2.sbatch
##
## usually takes 4 min
##
## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                        # Use 1 node
#SBATCH -t 2:00:00                                  # Set 2 hour time limit
#SBATCH -p prod                                     # Submit to the production 'partition'
#SBATCH -C "cpu"                                    # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                 # to allocate whole node
#SBATCH --account=proj83                            # your project number
#SBATCH --mem=0                                     # allocate entire memory to the job
#SBATCH --constraint=volta                          # setup node with GPU
#SBATCH --gres=gpu:1                                # setup one GPU
#SBATCH -o ./cluster/logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# setup python 3.9 environment
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/envs/npx_10m_384ch_unit_classes/bin/activate

# add custom package to python path and run pipe
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
export PYTHONPATH=$(pwd)
srun -n 1 python3.9 /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/src/pipes/postpro/horvath_vivo/save_metadata.py vivo_horvath probe_2