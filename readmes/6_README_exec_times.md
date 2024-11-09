
# Execution times

author: steeve.laquitaine@epfl.ch

## Preprocessing

### Fitting

* duration: 20 min per layer
* example pipeline: ./cluster/prepro/fitting/horvath/fit_silico_probe1_layer1.sbatch

### Wiring, Filtering, CMR

#### Neuropixels probe

* **Model (Biophy. spontaneous)**:    

    * duration: 1h20 (full recording)
        * gain + noise: 15 min
        * wiring + saving: 25 mins
        * filtering + cmr + saving: 35 mins

* **Model (Synthetic)**:  

    * duration: 21 mins

#### Denser probe

* **Horvath**:

    * duration: 15 min per depth
    * pipelines: 
        * ./cluster/prepro/horvath_vivo/process_probe1.sbatch
        * ./cluster/prepro/horvath_vivo/process_probe2.sbatch
        * ./cluster/prepro/horvath_vivo/process_probe3.sbatch

* **Model (Biophy spontaneous)**:

    * duration: 1h per depth    
    * example pipeline: 
        * ./cluster/prepro/horvath_silico/process_probe1.sbatch
        * ./cluster/prepro/horvath_silico/process_probe2.sbatch
        * ./cluster/prepro/horvath_silico/process_probe3.sbatch

## Validation

### Traces

### ANR

_amplitude-to-noise-ratio_

* duration for 10 minutes of recording: 6 min
* duration for the full recording

### PSD

_Power spectral density_

* neuropixels entire recordings: 5 mins (all at once on 4 nodes)

## Sorting

* Neuropixels
* 10 minutes of recording (08.08.2024)

spike sorter | npx vivo | npx spont | npx evoked| npx synth |
-------------|----------|-----------| ----------|-----------|
hs           | 9 min    | 11 min    | 6 min     | 11 min    |
ks           | 41 min   | 60 min    | 32 min    | 42 min    |
ks2          | 36 min   | 48 min    | 15 min ^  | 40 min    |
ks2.5        | 37 min   | 60 min    | 33 min    | 54 min    |
ks3          | 58 min   | 54 min    | 85 min    | 50 min    |
ks4          | 9 min    | 60 min    | 71 min    | 15 min    |

^ estimation for 10 min for npx evoked as ks2. Sorting worked only for max 4 min.

* Dense probe depth 1
* 10 minutes of recording

spike sorter | dense vivo | dense spont | 
-------------|------------|-----------| 
hs           | 8 min      | 
ks           | 10 min     |
ks2          | 
ks2.5        | 
ks3          | 
ks4          | 