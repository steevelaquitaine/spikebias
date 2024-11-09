#!/bin/bash -l
module purge; # load matlab after clean up, compile cuda code and setup python3.9 interpreter via spack env
module load matlab/r2019b;
cd /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/sorters/Kilosort3/CUDA;
matlab -batch mexGPUall;
module load spack;
cd /gpfs/bbp.cscs.ch/project/proj68/home/laquitai/sfn_2023/;
. /gpfs/bbp.cscs.ch/ssd/apps/bsd/2023-02-23/spack/share/spack/setup-env.sh;
spack env activate python3_9 -p;
source /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/3_sfn_2023/envs/npx_10m_384ch_unit_classes/bin/activate;
python3.9 -m src.pipes.controls.2_check_traces_in_ks3; # run pipeline