#!/bin/bash -l
#SBATCH -N 1                                        # Use 1 node
#SBATCH -t 03:00:00                                 # Set 1 hour time limit
#SBATCH -p prod                                     # Submit to the production 'partition'
#SBATCH -C "cpu"                                    # Constraint the job to run on nodes without SSDs.
#SBATCH --exclusive                                 # to allocate whole node
#SBATCH --account=proj68                            # your project number
#SBATCH --mem=0                                     # allocate entire memory to the job
#SBATCH --constraint=volta                          # setup node with GPU
#SBATCH --gres=gpu:1                                # setup one GPU
#SBATCH -o ./logs/cluster/output/slurm_jobid_%A_%a.out   # set log output file path
#SBATCH -e ./logs/cluster/output/slurm_jobid_%A_%a.err   # set log error file path

module load matlab
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/sorters_packages/Kilosort2/CUDA
matlab -batch mexGPUall
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/
module load spack
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2022-01-10/spack/share/spack/setup-env.sh
spack env activate spack_env -p
spack load python@3.9.7
source test_kilosort2_env/bin/activate
srun -n 1 python3.9 /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/spike-sorting/src/pipes/sorting/tests/test_kilosort2_and_3_vs_truth.py --conf 2020