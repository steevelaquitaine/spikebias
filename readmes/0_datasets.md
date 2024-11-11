
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

  - recording:
    - path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/raw/traces.pkl  (TO DELETE)
    - size: 119G
  - spikes:
  - nwb path (both): /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/neuropixels/concatenated_campaigns/nwb/npx_biophy_spont.nwb

**Evoked**

  - recording:
    - path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataeng/0_silico/4_spikesorting_stimulus_test_neuropixels_8-1-24__8slc_80f_360r_50t_200ms_1_smallest_fiber_gids/0fcb7709-b1e9-4d84-b056-5801f20d55af/campaign/raw/traces.pkl (TO DELETE)
    - size: 105G
  - spikes:
  - nwb path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/dataeng/0_silico/4_spikesorting_stimulus_test_neuropixels_8-1-24__8slc_80f_360r_50t_200ms_1_smallest_fiber_gids/0fcb7709-b1e9-4d84-b056-5801f20d55af/campaign/nwb/npx_biophy_evoked.nwb  

### Synthetic Buccino (2020)

  - recording:
    - nwb path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/raw_buccino/sub-MEAREC-250neuron-Neuropixels_ecephys.nwb'
    - size: 28G

## DENSE PROBE

### Horvath 

  - path: 
  - size: 

### Biophysical simulation

- recordings:

    - depth 1:
      - raw path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe1_hex0_rou04_pfr03_20Khz/raw/traces.pkl
      - nwb path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe1_hex0_rou04_pfr03_20Khz/nwb/dense_biophy_spont1.nwb'
        - size: 14 GB
    
    - depth 2:
      - raw path : /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe2_hex0_rou04_pfr03_20Khz/raw/traces.pkl
      - nwb path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe2_hex0_rou04_pfr03_20Khz/nwb/dense_biophy_spont2.nwb'
        - size: 9.3 GB

    - depth 3:
      - raw path: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe3_hex0_rou04_pfr03_20Khz/raw/traces.pkl
      - nwb path: '/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe3_hex0_rou04_pfr03_20Khz/nwb/dense_biophy_spont3.nwb'      
        - size: 15 GB      

- spikes:

    - path depth 1: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe1_hex0_rou04_pfr03_20Khz/raw/spiketrains.pkl
    - path depth 2: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe2_hex0_rou04_pfr03_20Khz/raw/spiketrains.pkl
    - path depth 3: /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/4_preprint_2023/0_silico/horvath/concatenated_campaigns/probe3_hex0_rou04_pfr03_20Khz/raw/spiketrains.pkl

## Auxiliary:

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

- Metadata files:
  - Atlas metadata: codebase repo at assets/metadata/ or load the open source Atlas: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QREN2T
  - Filtered_cells metadata (cells within 50 microns): assets/metadata/