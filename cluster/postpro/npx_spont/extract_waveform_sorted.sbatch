#!/bin/bash -l

##   author: steeve.laquitaine@epfl.ch
##     date: 19.10.2023
## modified: 15.01.2023
##
##    Usage:
##
##      sbatch cluster/postpro/marques_silico/extract_waveform_sorted.sbatch
##
## takes 23 min
##
## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                        # Use 1 node
#SBATCH -t 8:00:00                                  # Set 8 hour time limit
#SBATCH -p prod                                     # Submit to the production 'partition'
#SBATCH -C "cpu"                                    # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                 # to allocate whole node
#SBATCH --account=proj83                            # your project number
#SBATCH --mem=0                                     # allocate entire memory to the job
#SBATCH --constraint=volta                          # setup node with GPU
#SBATCH --gres=gpu:1                                # setup one GPU
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=laquitainesteeve@gmail.com

# install python 3.9 binary via spack and activate python env
module load spack
cd /gpfs/bbp.cscs.ch/project/proj85/home/laquitai/spikebias/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/envs/npx_10m_384ch_unit_classes/bin/activate

# run pipeline
srun -n 1 python3.9 -m src.pipes.postpro.marques_silico.extract_waveform_sorted