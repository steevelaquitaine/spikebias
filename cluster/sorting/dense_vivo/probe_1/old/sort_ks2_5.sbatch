#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 20.07.2023
## modified: 15.01.2023
##
## usage:
##
##      sbatch cluster/sorting/horvath_vivo/probe_1/sort_ks2_5.sbatch
##
## stats: 21 mins

## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 02:00:00                                      # Set 1 hour time limit
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

# clean up
module purge 
module use --append ~/modulefiles/
module load matlab2022toolboxes
cd /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/sorters/Kilosort-2.5/CUDA
matlab -batch mexGPUall
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/npx_10m_384ch_unit_classes/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.horvath_vivo.probe_1.sort_ks2_5



