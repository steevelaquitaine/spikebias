
# Spike sort with RTX5090 GPU

Tested on Ubuntu 24 with RTX 5090 GPU

Two methods:

1. From Notebook
    1. Enable forward compatibility if your GPU and CUDA libraries are more recent and not supported by editing your matlab `startup.m` file to contain "parallel.gpu.enableCUDAForwardCompatibility(true)" and:

        ```bash
        # manually compile Kilosort3 with CUDA support for forward compatibility
        sudo apt install gcc-11 g++-11 # install gcc 11 compiler
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 # enable temporary
        cd /home/steeve/steeve/epfl/code/spikebias/dataset/01_intermediate/sorters/Kilosort3_buttw_forwcomp/CUDA/
        matlab -batch mexGPUall  # compile matlab mex files
        ```

    2. Run notebook ss.run_sorter should work.


2. From terminal, modifying spikeinterface's created bash script. You also also run spike sorting manually by replacing SpikeInterface's "run_kilosort.sh" script (created by ss.run_sorter) with and run: 

  ```bash
  #/bin/bash -->
  #manually compile Kilosort3 with CUDA support for forward compatibility
  sudo apt install gcc-11 g++-11 # install gcc 11 compiler
  sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 # enable temporary

  cd /home/steeve/steeve/epfl/code/spikebias/dataset/01_intermediate/sorters/Kilosort3_buttw_forwcomp/CUDA/
  matlab -batch mexGPUall  # compile matlab mex files

  run Kilosort3 with forward compatibility activated for that session
  cd "/home/steeve/steeve/epfl/code/spikebias/SortingKS3/sorter_output"

  matlab -nosplash -nodisplay -r "parallel.gpu.enableCUDAForwardCompatibility(true); kilosort3_master('/home/steeve/steeve/epfl/code/spikebias/SortingKS3/sorter_output', '/home/steeve/steeve/epfl/code/spikebias/dataset/01_intermediate/sorters/Kilosort3_buttw_forwcomp')"
  ```