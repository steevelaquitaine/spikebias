# Datasets

author: steeve.laquitaine@epfl.ch; laquitainesteeve@gmail.com

* Config. must be edited in: 
  * `src/nodes/utils.py`
  * `src/nodes/dataeng/lfp_only/campaign_stacking.py`

TODO:

- get all data from blueconfig to skip blueconfig [DOING]
- make all NWB files of raw wired recordings and upload to DANDI archive [DOING]
- keep only the electrode location from the weights - much lighter dataset.
- remove all hard coded paths of the codebase (mostly in /pipes)

## NEUROPIXELS 

### Marques-Smith (2023)

  - type: binary
  - path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/vivo_rat_cortex_marques/c26/c26_npx_raw-001.bin
  - size: 28G

### Biophysical simulations

**Spontaneous**

  - RecordingExtractor and Ground truth SortingExtractor files:
    - path: 001250/sub-001/sub-001_ecephys.nwb
    - size: 109.9 GB (downloading takes 20 hours)
  - Electrode and cell metadata files: spikebias/assets/metadata/silico_neuropixels/npx_spont/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

**Evoked**

  - RecordingExtractor and Ground truth SortingExtractor files:
    - path: 001250/sub-002/sub-002_ecephys.nwb
    - size: 97.6 GB (downloading takes 23 hours)
  - Electrode and cell metadata files: spikebias/assets/metadata/silico_neuropixels/npx_evoked/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

### Synthetic Buccino (2020)

  - recording:
    - nwb path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/raw_buccino/sub-MEAREC-250neuron-Neuropixels_ecephys.nwb'
    - size: 28G

## DENSE PROBE

### Horvath 

  - path: 
  - size: 

### Biophysical simulation

- DEPTH 1:
  - RecordingExtractor and Ground truth SortingExtractor files:
    - path: 001250/sub-003/sub-003_ecephys.nwb
    - size: 14.6 GB (downloading takes < 8 hours)
  - Electrode and cell metadata files: spikebias/assets/metadata/dense_spont/probe_1/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

- DEPTH 2:
  - RecordingExtractor and Ground truth SortingExtractor files:
    - path: 001250/sub-004/sub-004_ecephys.nwb
    - size: 9.9 GB (downloading takes < 8 hours)
  - Electrode and cell metadata files: spikebias/assets/metadata/dense_spont/probe_2/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

- DEPTH 3:
  - RecordingExtractor and Ground truth SortingExtractor files:
    - path: 001250/sub-005/sub-005_ecephys.nwb
    - size: 15.6 GB (downloading takes < 8 hours)
  - Electrode and cell metadata files: spikebias/assets/metadata/dense_spont/probe_3/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

## Auxiliary

- Weight files:

  - neuropixels probe:
    - path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataset/weights/coeffsneuropixels.h5'
    - size: 227 GB

  - dense probe:
    - depth 1:
      - path : /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataset/weights/coeffsdenseprobe1.h5
      - size: 77 GB
    - depth 2:
      - path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataset/weights/coeffsdenseprobe1.h5
      - size: 77 GB
    - depth 3:
      - path : /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataset/weights/coeffsdenseprobe1.h5
      - size: 77 GB

  - Atlas (optional): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QREN2T