#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
##     date: 11.02.2023
## modified: 14.02.2023
##
## usage:
##
##      sbatch cluster/sorting/others/spikewarp/s2X/d10m/ks.sh
##
## stats: ??

## Setup job config

#!/bin/bash -l
#SBATCH -J ks-spikewarp2X-10m                            # name
#SBATCH -N 1                                             # Use 1 node
#SBATCH -t 08:00:00                                      # Set 8 hour time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH --constraint=volta                               # setup node with 16 GB RAM GPU (min required 8 GB)
#SBATCH --gres=gpu:1                                     # setup one GPU
#SBATCH -o ./cluster/logs/cluster/output/sorting/slurm_nm_%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/sorting/slurm_nm_%x_id_%A.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# clean up, load matlab, check that matlab can find GPU, compile cuda code and setup python3.9 env. via spack env
module purge 
module use --append ~/modulefiles/
module load matlab2022toolboxes
matlab -r "disp(ver); disp('Can matlab find GPU:'); disp(getenv('HOSTNAME')); disp(gpuDevice(1)); exit;"
cd /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/sorters/KiloSort/CUDA
matlab -batch mexGPUall
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/preprint_2023/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.sorting.others.spikewarp.s2X.d10m.ks


