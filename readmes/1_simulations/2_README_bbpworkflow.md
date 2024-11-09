
# README bbp-workflow

author: steeve.laquitaine@epfl.ch

### Install bbp-workflow

```bash
cd ~
python3 -m venv bbp_workflow_env
source bbp_workflow_env/bin/activate
pip3 install --upgrade -i https://bbpteam.epfl.ch/repository/devpi/simple bbp-workflow-cli
module load unstable py-bbp-workflow-cli
bbp-workflow version 
```

### Move campaign(experiment) files to another project path

0. get your sim-config-url (see `README_nexis.md`)
1. Use MoveSimCampaigns:

```bash
# create the destination path;
# move custom folders;
# move campaign;
mkdir /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/neuropixels_lfp_10m_32ch_hex0_rou04_pfr03_10Khz_2023_10_01/;
rsync -rv /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/neuropixels_lfp_10m_32ch_hex0_rou04_pfr03_10Khz_2023_10_01/bbp_workflow_env /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/neuropixels_lfp_10m_32ch_hex0_rou04_pfr03_10Khz_2023_10_01/;
rsync -rv /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/neuropixels_lfp_10m_32ch_hex0_rou04_pfr03_10Khz_2023_10_01/workflows /gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/neuropixels_lfp_10m_32ch_hex0_rou04_pfr03_10Khz_2023_10_01/;
bbp-workflow launch --follow \
    bbp_workflow.simulation MoveSimCampaign \
        sim-config-url=https://bbp.epfl.ch/nexus/v1/resources/bbp/somatosensorycortex/_/2f69d39b-51ff-4a30-ae8a-934347c45698 \
        path-prefix=/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/neuropixels_lfp_10m_32ch_hex0_rou04_pfr03_10Khz_2023_10_01;
```

2. If you have absolute paths in your BlueConfig (you set them in Generate... in BBPworkflow), run this bash script to rename the paths, in each campaign's path:

```bash
# fix_paths.sh
rsed() {   [[ -z $2 ]] && echo "usage: ${FUNCNAME[0]} oldtext newtext" && return;   command find . -type f -exec sed -i "s+${1}+${2}+g" {} \;; }

OLD_PATH=/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/test_neuropixels_lfp_10m_384ch_hex_O1_40Khz_2023_08_17/421db120-c09a-4b21-9b5d-f63e2c0d15b4/
NEW_PATH=/gpfs/bbp.cscs.ch/project/proj85/scratch/laquitai/raw/test_neuropixels_lfp_10m_384ch_hex_O1_40Khz_2023_08_17/421db120-c09a-4b21-9b5d-f63e2c0d15b4/

cd $NEW_PATH

for i in {0..11}
do
 cd $i
 rsed $OLD_PATH $NEW_PATH
 cd ..
done
```

run in terminal: 
```
bash fix_paths.sh
```

### References

(1) https://bbp.epfl.ch/documentation/projects/bbp-workflow/latest/generated/bbp_workflow.simulation.task.html#bbp_workflow.simulation.task.MoveSimCampaign  
(2) https://bbp.epfl.ch/nexus/v1/resources/bbp/somatosensorycortex/_/2f69d39b-51ff-4a30-ae8a-934347c45698  