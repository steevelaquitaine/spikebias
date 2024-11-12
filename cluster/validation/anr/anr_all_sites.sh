#!/bin/bash -l

## author: steeve.laquitaine@epfl.ch
## Usage:
##
##      sbatch cluster/validation/anr/anr_all_sites.sh
##
## config

#SBATCH --job-name="anr_full_all_sites"
#SBATCH --partition=prod
#SBATCH --nodes=9
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=72
#SBATCH --account=proj85
#SBATCH --exclusive
#SBATCH --constraint=cpu
#SBATCH --mem=0
#SBATCH -o ./cluster/logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

module purge 
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/spikinterf0_100_5/bin/activate
srun python3.9 -m src.pipes.validation.anr.anr_all_sites