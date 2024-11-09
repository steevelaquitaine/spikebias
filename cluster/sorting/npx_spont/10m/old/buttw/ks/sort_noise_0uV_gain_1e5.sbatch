#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 20.07.2023
## modified: 15.01.2023
##
## usage:
##
##      sbatch cluster/sorting/marques_silico/10m/buttw/ks/sort_noise_0uV_gain_1e5.sbatch
##
## stats: 21 mins

## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 1 hour time limit
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
module purge 
module use --append ~/modulefiles/
module load matlab2022toolboxes
cd /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/sorters/KiloSort/CUDA
matlab -batch mexGPUall
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.marques_silico.10m.buttw.ks.sort_noise_0uV_gain_1e5