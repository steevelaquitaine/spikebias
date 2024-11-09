
# README - SPIKEWARP

author: steeve.laquitaine@epfl.ch; laquitainesteeve@gmail.com

## Requirements 

This are required for the preprocessing and some dowstream analyses.

## Datasets

- Weight file: 
    - path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataset/weights/coeffsneuropixels.h5'
    - size: 227G

## Configuration

1. Add experiment path to `load_locations_from_weights` in `src/nodes/dataeng/lfp_only/campaign_stacking.py`

## Preprocessing (1h20)

* Concatenate simulations
* Noise and amplitude fitting
* Wiring, filtering, common median referencing

```bash
sbatch cluster/prepro/others/spikewarp/sparse/1X/process.sh
```