# Datasets

author: steeve.laquitaine@epfl.ch; laquitainesteeve@gmail.com

* Datasets configuration must be edited in two nodes of the codebase: 
  * `src/nodes/utils.py`
  * `src/nodes/dataeng/lfp_only/campaign_stacking.py`

## NEUROPIXELS

### Marques-Smith et al., (2023)

  - type: binary
  - path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/vivo_rat_cortex_marques/c26/c26_npx_raw-001.bin
  - size: 28G

### Biophysical simulations

There are metadata describing each ground truth unit in the raw .nwb file not the fitted one.

**Spontaneous**

  - RecordingExtractor and Ground truth SortingExtractor files:
    - raw:
      - path: 001250/sub-001/sub-001_ecephys.nwb
      - size: 109.9 GB (downloading takes 20 hours)
    - fitted:
      - path: 001250/sub-001-fitted/sub-001-fitted_ecephys.nwb
      - size: 117.4 GB (downloading takes 20 hours)

  - Electrode and cell metadata files: spikebias/assets/metadata/silico_neuropixels/npx_spont/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

**Spontaneous disconnected**


**Evoked (20 kHz)**

  - RecordingExtractor and Ground truth SortingExtractor files:
    - raw
      - path: 001250/sub-002/sub-002_ecephys.nwb    
      - size: 97.6 GB (downloading takes 23 hours)
    - fitted
      - path: 001250/sub-002-fitted/sub-002-fitted_ecephys.nwb    
      - size: 103.2 GB (downloading takes 23 hours)

  - Electrode and cell metadata files: spikebias/assets/metadata/silico_neuropixels/npx_evoked/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

**Evoked (40 kHz)**
  
  - RecordingExtractor and Ground truth SortingExtractor files:
    - raw
      - path: 001250/sub-002/sub-002_ecephys.nwb
      - size: 97.6 GB (downloading takes 23 hours)

  - Electrode and cell metadata files: spikebias/assets/metadata/silico_neuropixels/npx_evoked_40Khz/
    - cell_properties.h5
    - filtered_cells.npy
    - layers.npy
    - regions.npy

### Synthetic Buccino et al.,　(2020)

  - recording:
    - nwb path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/raw_buccino/sub-MEAREC-250neuron-Neuropixels_ecephys.nwb'
    - size: 28G

## DENSE PROBE

### Horvath et al., 

Raw from https://doi.org/10.6084/m9.figshare.14555421: 

- horvath/Rat01/Insertion1/Depth1/Rat01_Insertion1_Depth1.nwb
- horvath/Rat01/Insertion1/Depth1/Rat01_Insertion1_Depth2.nwb
- horvath/Rat01/Insertion1/Depth1/Rat01_Insertion1_Depth3.nwb

### Spontaneous biophysical simulation

There are metadata describing each ground truth unit in the raw .nwb file not the fitted one.

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

## REYES PROBE

- Summed extracellular recording
  - RecordingExtractor and SortingExtractor file:
    - path: sub-biophy-summed-traces-reyes/sub-biophy-summed-traces-reyes_ses-006_ecephys
  
- Isolated extracellular recording
  - RecordingExtractor and SortingExtractor file:
    - path: sub-biophy-isolated-traces-reyes/sub-biophy-isolated-traces-reyes_ses-006_ecephys

- Cell 3754013 morphology metadata:
  - spikebias/assets/morph_cell_3754013.h5

## Auxiliary data

- Weight files:

  - neuropixels probe:
    - path: '/gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/dataset/weights/coeffsneuropixels.h5'
    - path (doing): /gpfs/bbp.cscs.ch/project/proj85/laquitai/preprint_2024/dataset/weights/coeffsneuropixels.h5
    - size: 227 GB

  - dense probe:
    - depth 1:
      - path : /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/dataset/weights/coeffsdenseprobe1.h5
      - path (doing): /gpfs/bbp.cscs.ch/project/proj85/laquitai/preprint_2024/dataset/weights/coeffsdenseprobe1.h5
      - size: 77 GB
    - depth 2:
      - path: /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/dataset/weights/coeffsdenseprobe1.h5
      - size: 77 GB
    - depth 3:
      - path : /gpfs/bbp.cscs.ch/project/proj85/laquitai/spikebias_paper/dataset/weights/coeffsdenseprobe1.h5
      - size: 77 GB

  - Atlas (optional): https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QREN2T

* O1 sonata circuit: https://zenodo.org/records/11113043 (51 GB)
* full sonata circuit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HISHXN **(300 GB)**
    * this is the largest circuit which includes the cortical column cells used for all biophysical simulations
* note: cell id in the sonata circuit is the cell id in the old circuit - 1.
* all the cell properties can be retrieved from these circuits.