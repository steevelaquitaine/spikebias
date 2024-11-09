
# BBP WORKFLOW 

# Installation

Create virtual environment:

```bash
# move to your experiment path:
cd /gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/raw/neuropixels_lfp_10m_384ch_2023_06_19

# create and setup virtual environment (python3.6 works)
python3 -m venv bbp_workflow_env
source bbp_workflow_env/bin/activate

# install cli
pip3 install --upgrade -i https://bbpteam.epfl.ch/repository/devpi/simple bbp-workflow-cli

# load module if on BB5 cluster
module load unstable py-bbp-workflow-cli

# check it is working (will ask gaspar password)
bbp-workflow version 
# Version tag: latest
# Client version: 3.1.0.dev2
# Server version: 3.1.14
```

* see (1), official documentation (2)

# Configuration

What should I enter for these paraneters: ask `Genrish` (slack):  

0. Edit your BlueConfig template : 

in `BlueConfig__GroupwiseConductanceDepolarisation__SSCx-O1_NoStim.tmpl`.

1. In `GenerateCampaign__GroupwiseConductanceDepolarisation__SSCx-O1`:
    
    1. edit `account: proj83` to an account you have access to

    2. edit `coords:` values (variable parameters accessible in via `bluetl`) under [GenerateSimulationCampaign]:
            
        ```
        coords: {
                "depol_stdev_mean_ratio": [0.4],
                "ca": [1.05],
                "desired_connected_proportion_of_invivo_frs": [0.3],
                "seed": [1,2,3,4,5,6,7,8,9,10,11,12]
                }
        ```

    2. Edit `attrs:` values (fixed parameters):
        1. edit `path_prefix:` value under `attrs: {..}`:

        ```
        "path_prefix": "/gpfs/bbp.cscs.ch/project/proj68/scratch/raw/reyes_probe_lfp_2023_02_06"
        ```
        
        2. edit `name:` as you want e.g., `SSCx-Bio_M-2023-02-08-reyes-lfp`

        3. Edit `sim_duration:` e.g., to 50000 (50 secs).

2. To initialize the campaign, run this from workflows' parent directory:

This runs the `MODULE` bbp_workflow.simulation and `TASK` specified under the [GenerateSimulationCampaign] section:

    ```bash
    # note that this is a single command
    bbp-workflow launch-bb5 --follow --config bbp_workflow_config/GenerateCampaign__GroupwiseConductanceDepolarisation__SSCx-O1_NoStim.cfg bbp_workflow.simulation GenerateSimulationCampaign    
    ```

3. Edit the launch script `LaunchCampaign__GroupwiseConductanceDepolarisation__SSCx-O1`
    1. Edit `account: proj68`: 

    2. Edit the url: replace it with the url produced by `GenerateCampaign..cfg` (if it is successful: produces a smiley face)) in the file `LaunchCampaign__GroupwiseConductanceDepolarisation__SSCx-O1` after the key `sim-config-url:`

* It is printed in the terminal after: `Simulation campaign:` and look like this `https://bbp.epfl.ch/nexus/v1/resources/bbp/somatosensorycortex/_/e5b6b059-9d42-46ff-8a19-9cd858c536c2`

    ```
    sim-config-url: https://bbp.epfl.ch/nexus/v1/resources/bbp/somatosensorycortex/_/3f543462-7a6f-4742-b62e-f03412769dbd
    ```

5. Run the command below from the parent (`workflows/`):
    Note that the parent must be called `workflows`.
    
    ```bash
    bbp-workflow launch --follow --config workflows/LaunchCampaign__GroupwiseConductanceDepolarisation__SSCx-O1.cfg bbp_workflow.simulation SimulationCampaign
    ```

6. Track status

* Click on the printed link: e.g., https://bbp-workflow-laquitai.kcp.bbp.epfl.ch/static/visualiser/index.html#order=4%2Cdesc


# Documentations

On confluence.

* main docs:
    * (3,2,1)
* Example configs:
    * see files in `Desktop/SpontaneousScanExample/` folder provided by James Ibsister.
    * see files in `Desktop/3-ThalamicStimuli-WhiskerFlickExample-18-01-23/` folder provided by Joseph Tharayil.

# Troubleshooting

**Issue**: you need to have access permission to `METypePath`:  
    * Solution: replace `METypePath /gpfs/bbp.cscs.ch/project/proj45/scratch/S1full/METypes` by `METypePath /gpfs/bbp.cscs.ch/project/proj68/scratch/METypes`


# Notes

* `sim_duration`: is set in `GenerateCampaign ..cfg` file and called in `BlueConfig...tmpl` .. file
* `param-processors`: this section is needed. It lists a python module `GenerateCampaign_ParamProcessors` that should exist in the path.
    * `lookup_projection_locations`: is a python dependency of GenerateCampaign_ParamProcessors and should exist in the path.
    * `flatmap_utility`: is a python dependency
    * `stimulus_generation`: is a python dependency
    * `spikewriter`: is a python dependency

* Be careful of commas. Remove any comma at the end of a list, otherwise .cfg files' parsing fails.

# Useful commands 

```bash
scp -r Desktop/my_campaign laquitai@bbpv1.epfl.ch:/gpfs/bbp.cscs.ch/project/proj68/scratch/laquitai/bbp_workflow/neuropixels-lfp-2023-02-06/
```

# References

(1) https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/Workflow  
(2) https://bbp.epfl.ch/documentation/projects/bbp-workflow/latest/generated/bbp_workflow.simulation.task.html#bbp_workflow.simulation.task.CortexNrdmsPySim
(3) https://bbpteam.epfl.ch/project/spaces/display/BBPNSE/How+to+launch+Simulation+Campaigns#HowtolaunchSimulationCampaigns-simurl  