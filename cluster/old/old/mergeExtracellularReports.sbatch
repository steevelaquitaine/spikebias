#!/bin/bash -l
#SBATCH --job-name="mergeExtracellularReports"
#SBATCH --partition=prod
#SBATCH --nodes=30
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=20
#SBATCH --account=proj68
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --mem=0
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path

module load unstable hpe-mpi/2.25.hmpt matlab
module load spack
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
spack env activate spack_env -p
spack load python@3.9.7
srun -n 30 python3 app.py simulation merge --conf $conf_date