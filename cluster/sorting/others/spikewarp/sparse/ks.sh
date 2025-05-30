#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 20.07.2023
## modified: 15.01.2023
##
## usage:
##
##      sbatch cluster/sorting/others/spikewarp/sparse/ks.sh
##
## stats: 21 mins

## Setup job config

#!/bin/bash -l
#SBATCH -J ks-spikewarp-sparse-10m                           # job name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 1 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with GPU
#SBATCH --gres=gpu:1                                     # setup one GPU
#SBATCH -o ./cluster/logs/cluster/output/spikewarp/sorting/sparse/%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/spikewarp/sorting/sparse/%x_id_%A.err   # set log error file path
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
matlab -r "disp(ver); disp('Can matlab find GPU:'); disp(getenv('HOSTNAME')); disp(gpuDevice(1)); exit;"
cd /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/sorters/KiloSort/CUDA
matlab -batch mexGPUall
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/envs/spikinterf0_100_5/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.others.spikewarp.sparse.ks