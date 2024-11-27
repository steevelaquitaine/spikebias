#!/bin/bash -l

##   author: steeve.laquitaine@epfl.ch
##     date: 03.11.2023
## modified: 03.11.2023
##
##    Usage:
##
##      sbatch cluster/postpro/label_sorted_unit_layer_silico_horvath.sbatch
##
## usually takes 30 min
##
## Setup job config

#!/bin/bash -l
#SBATCH -N 1                                        # Use 1 node
#SBATCH -t 2:00:00                                  # Set 2 hour time limit
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
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/sfn_2023/   
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2024-02-01/spack/share/spack/setup-env.sh
spack env activate python3_9 -p
source /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/envs/npx_10m_384ch_unit_classes/bin/activate

# extract spike waveforms
srun -n 1 python3.9 -m src.pipes.postpro.waveform_silico_horvath_1
srun -n 1 python3.9 -m src.pipes.postpro.waveform_silico_horvath_2
srun -n 1 python3.9 -m src.pipes.postpro.waveform_silico_horvath_3

# infer nearest contact and layer
srun -n 1 python3.9 -m src.pipes.postpro.get_silico_horvath_sorted_units_contact_and_layer

# add layer to SortinExtractor
srun -n 1 python3.9 -c "from src.nodes.postpro.layer import label_sorted_unit_layer_horvath_silico; label_sorted_unit_layer_horvath_silico()"