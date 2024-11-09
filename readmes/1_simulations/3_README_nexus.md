
# README_nexus.md  


### Access the list of your run campaigns


1. Open (1) in your browser to go to NEXUS
2. Connect

### Get sim-config-url

1. Via NEXUS (see above)
2. Via the stored run campaign data
    1. Go to data directory (e.g., /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/neuropixels_lfp_10m_32ch_hex0_rou04_pfr03_10Khz_disconnected_2023_10_01/2f69d39b-51ff-4a30-ae8a-934347c45698)
    2. Open `config.json` and copy hash under the `name` key (e.g., 2f69d39b-51ff-4a30-ae8a-934347c45698)
    3. Concatenate as `https://bbp.epfl.ch/nexus/v1/resources/bbp/somatosensorycortex/_/2f69d39b-51ff-4a30-ae8a-934347c45698`

### References

(1) https://bbp.epfl.ch/nexus/web/