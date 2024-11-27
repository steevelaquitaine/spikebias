#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 20.07.2023
## modified: 15.01.2023
##
## usage:
##
##       sbatch cluster/sorting/others/spikewarp/sort_ks2_5.sbatch
##
## stats: 21 mins
##
## note: Use envs with spikeinterface 0.96.1 that worked else, unrecognized dshift error

## Setup job config

#!/bin/bash -l
#SBATCH -J ks25-spikewarp-10m                           # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 02:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # setup one GPU
#SBATCH -o ./cluster/logs/cluster/output/spikewarp/sorting/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/spikewarp/sorting/slurm_jobid_%A_%a.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# clean up
module purge 

# load matlab after clean up, compile cuda code and setup python3.9 interpreter via spack env
# module load matlab/r2019b 
# module load matlab/r2022b
# Load matlab path with toolboxes: "/gpfs/bbp.cscs.ch/apps/tools/matlab/r2022b-addons/bin/matlab"
# 1. create the module file matlab2022toolboxes below with vim once:
# mkdir -p ~/modulefiles/
# ```text
# %Module
# set basedir "/gpfs/bbp.cscs.ch/apps/tools/matlab/r2022b-addons"
# prepend-path PATH "${basedir}/bin"
# prepend-path LD_LIBRARY_PATH "${basedir}/lib64"
# ```
# 2. append your matlab module to available modules
module use --append ~/modulefiles/
module load matlab2022toolboxes
cd /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/sorters/Kilosort-2.5/CUDA
matlab -batch mexGPUall
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/preprint_2024/envs/spikinterf0_100_5/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.others.spikewarp.ks2_5
