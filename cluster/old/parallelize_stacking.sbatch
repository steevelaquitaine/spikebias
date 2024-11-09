#!/bin/bash -l

## author: 
##      steeve.laquitaine@epfl.ch
##
## Usage:
##
## In python: 
##      os.system('sbatch sbatch/parallelize_chunking.sbatch 2023_01_13')
##
## In the terminal:
##      edit $conf_date with a simulation conf: e.g., 2023_01_13 then run: 
##      sbatch sbatch/parallelize_stacking.sbatch

## Setup job config

#SBATCH --job-name="parallelize_stacking"
#SBATCH --partition=prod
#SBATCH --nodes=11
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=20
#SBATCH --account=proj68
#SBATCH --exclusive
#SBATCH --constraint=clx
#SBATCH --mem=0
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path

module load unstable hpe-mpi/2.25.hmpt matlab
module load spack
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
spack env activate spack_env -p
spack load python@3.9.7

## run script
# TODO: 
# - variabilize --conf
## srun -n 10 python3 /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/src/pipelines/simulation/dataeng/chunking_parallelized.py --conf $conf_date
# srun -n 10 python3 /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/src/nodes/dataeng/silico/stacking_parallelized.py --exp supp/silico_reyes --conf 2023_01_13
srun -n 11 python3 /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/src/nodes/dataeng/silico/stacking_parallelized.py --exp silico_neuropixels --conf 2023_02_19