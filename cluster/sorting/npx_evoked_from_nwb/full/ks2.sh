#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 11.02.2023
## modified: 11.02.2023
##
## usage:
##
##      sbatch cluster/sorting/npx_evoked/full/ks2.sh
##
## stats: ??

## Setup job config

#!/bin/bash -l
#SBATCH -J ks2-evoked-full                               # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 8 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=gpu_32g                             # setup node with GPU with 32 GB RAM (2x more than before)
#SBATCH -o ./cluster/logs/cluster/output/npx_evoked/sorting/full/slurm_nm_%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/npx_evoked/sorting/full/slurm_nm_%x_id_%A.err   # set log error file path
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
# clean up, load matlab, check that matlab can find GPU, compile cuda code and setup python3.9 env. via spack env
module purge 
module use --append ~/modulefiles/
module load matlab2022toolboxes
matlab -r "disp(ver); disp('Can matlab find GPU:'); disp(getenv('HOSTNAME')); disp(gpuDevice(1)); exit;"
cd /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/sorters/Kilosort-2.0.2/CUDA # last version patched
matlab -batch mexGPUall
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/envs/spikinterf0_100_5/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.npx_evoked.full.ks2