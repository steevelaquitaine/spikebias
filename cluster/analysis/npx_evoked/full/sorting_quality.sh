#!/bin/bash -l

## author: steeve.laquitaine@epfl.ch
## Usage:
##
##      sbatch cluster/analysis/npx_evoked/full/sorting_quality.sh
##
## Takes 12 minutes
##
## config

#!/bin/bash -l
#SBATCH -J npx-evoked-quality-full                       # job name
#SBATCH --nodes=6                                        # Use 6 nodes
#SBATCH -t 08:00:00                                      # Set 8 hours time limit
#SBATCH -p prod                                          # Submit to the production 'partition'
#SBATCH -C "cpu"                                         # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                      # to allocate whole node
#SBATCH --account=proj83                                 # your project number
#SBATCH --mem=0                                          # allocate entire memory to the job
#SBATCH -o ./cluster/logs/cluster/output/analysis/quality/npx_evoked/full/slurm_nm_%x_id_%A.out   # set log output file path
#SBATCH -e ./cluster/logs/cluster/output/analysis/quality/npx_evoked/full/slurm_nm_%x_id_%A.err   # set log error file path
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
srun python3.9 -m src.pipes.analysis.npx_evoked.full.code.igeom
srun python3.9 -m src.pipes.analysis.npx_evoked.full.sorting_quality