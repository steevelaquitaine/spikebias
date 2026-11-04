# Spikebias

[![DOI](https://zenodo.org/badge/885915678.svg)](https://doi.org/10.5281/zenodo.17389967)

author: laquitainesteeve@gmail.com  
contributors: tharayiljoe@gmail.com; milo.imbeni@gmail.com; james.isbister@openbraininstitute.org; michael.reimann@openbraininstitute.org

# System requirements

All software dependencies and operating systems (including version numbers)
Versions: 

* Tested OS:
    * Ubuntu 24.04.1 LTS (32 cores, 188 GB RAM, Intel(R) Core(TM) i9-14900K ＠3.2 GHz/5.8 GHz); RTX590 GPU (40GB VRAM)
    * Ubuntu 24.04.1 LTS (16 cores, 62 GB RAM, Intel Xeon Platinum 8259CL ＠2.50GHz)
    * Mac OS X 10.15.7 (8 cores, 8 GB RAM, BuildVersion: 19H2026, Quad-Core Intel Core i7 ＠2.50GHz) 
* Tested Conda versions:
    * 24.11.3
    * 24.9.2
* Tested Software:
  * python 3.9.7
  * python 3.10.8 (demo)
  * the python dependencies are listed in envs/demo.yml for the demo and envs/spikebias.yml for the paper's analyses:
  * Matlab R2022b with: 
    * the Parallel Computing Toolbox (Version 7.7)
    * the Signal Processing Toolbox (Version 9.1) 
    * the Statistics and Machine Learning Toolbox (version 12.4)
  * the spike sorters tested are:
    * kilosort cloned from https://github.com/cortex-lab/KiloSort.git
    * kilosort 2 release: https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.0.2.tar.gz
    * kilosort 2.5 release https://github.com/MouseLand/Kilosort/archive/refs/tags/v2.5.tar.gz
    * kilosort 3 from spikeinterface 0.100.5
    * kilosort 4 from spikeinterface 0.100.5

Versions the software has been tested on: the versions referenced above.

Any required non-standard hardware: 
* most of the raw source code necessary to generate the small intermediate dataset were run on a supercomputer and require MPI and parallel computing on about ten computer nodes.
* spike sorting requires GPU.
* disk storage of about 2TB

# Installation guide

## Instructions

Create and activate the main python virtual environment (spikinterf0_100_5) to run spikebias:

1. move to the root path of the repository and create env with: 

```bash 
conda env create -f envs/spikebias.yml --prefix ./envs/spikebias # create
conda activate ./envs/spikebias # activate
```

2. Run a function from the custom package 

```bash
# move to repository path
import os
os.chdir("/home/jovyan/steevelaquitaine/spikebias/")

# import a module from the custom package
from src.nodes.utils import get_config
from src.nodes.validation import noise

# calculate mean absolute deviation of a recording trace
trace_mad = noise.torch_mad(trace)
```

## Typical install time

Conda install takes 30 minutes.

# Demo

See jupyter notebooks in notebooks/0_demo/

# Instructions for use 

How to run the software on the manuscript's data.

1. You can run the notebooks in notebooks/1_results and notebooks/2_supp_results with the installed virtual environment by running the command below, then selecting the kernel "spikebias" in the notebook:

```bash
python -m ipykernel install --user --name spikebias --display-name "spikebias"
```

# Datasets

* The full sonata circuit: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/HISHXN **(300 GB)**
    * this is the largest circuit which includes the cortical column cells used for all biophysical simulations
* The O1 sonata circuit: https://zenodo.org/records/11113043 (51 GB)
* The cell id in our experiments are the cell ids in the CELL_ID_SONATA - 1.
* All the cell properties can be retrieved from these circuits.

```bash
dataset/
    00_raw/ (505 GB)
```

# Models

Trained models are in `models/` (see `models/README.md`).

* `models/cebra/`: 100 cross-validated CEBRA models (`.pt` files, about 4.6 MB each) for each of three runs that classify the sorted single-units of Kilosort4 into 'good' and degraded units from their full spike waveforms and quality metrics (Supplementary Fig. 5c-e):
    * `spont_40_iterations/` (spontaneous regime, Supplementary Fig. 5c)
    * `evoked_40_iterations/` (evoked regime, Supplementary Fig. 5d)
    * `spont_100_iterations/` (spontaneous regime, training loss of Supplementary Fig. 5e)
* each folder also holds `folds.csv` (precision, recall, training and test unit indices and predictions of each fold) and `units.csv` (sorted unit id and label of each row of the dataset)
* the `.pt` weights are not tracked by git (see `.gitignore`)
* load a model (torch checkpoints: only load files that you trust) with `cebra==0.4.0` and `scikit-learn==1.5.2` (see `envs/cebraspike.txt`):

```python
from src.nodes.models.CebraSpike import utils
model = utils.load_crossval_model("models/cebra/spont_40_iterations", fold=0)
```

```bash
models/
    cebra/ (1.1 GB)
```

# [TODO]

- update the dense_probe1 SortingExtractors uploaded on dandiarchive (dandiset 001250, `sub-003-fitted`). The KS3, KS2.5, KS2, KS and HS 10-minute sortings (`sub-003-fitted_ses-ks3-10m.nwb`, `ses-ks25-10m`, `ses-ks2-10m`, `ses-ks-10m`, `ses-hs-10m`) are duplicates of the dense_probe2 sortings (`sub-004-fitted`, same ses-* files): same number of units and same spike trains. They do not match the dense_probe1 ground truth (`sub-003-fitted_ecephys.nwb`), so all well-sorted ground truth counts are about 0 for depth 1.
    - the correct dense_probe1 sortings are in `dataset/01_intermediate/sorting/dense_probe1/`. Number of units (KS4, KS3, KS2.5, KS2, KS, HS): 94, 238, 518, 512, 693, 29 (dandiarchive duplicates of dense_probe2: 131, 293, 530, 533, 498, 88)
    - the dandiarchive KS4 sorting of dense_probe1 is also a duplicate of dense_probe2 (131 units; the DANDI copy `dataset/01_intermediate/sorting/dense_probe1/SortingKS4_10m_` was removed from the working tree, it is still in git history (commit a6e5958); the correct one has 94 units, in `SortingKS4_10m`). The dandiarchive KS4 sortings of dense_probe2 and dense_probe3 (`SortingKS4_10m_`) are identical to the GitHub ones (all spike trains compared) and were removed from the working tree, they are still in git history (commit a6e5958)
- save spikecurate regression models in models/
- upload models to huggingface

## [TODO] Editor's revision requests, per figure

Source: `paper/NCOMMS-25-22328B_Laquitaine_Author_Checklist_1771884331_25.docx`. Source data and svg of each panel: `figures/8_source_data/<panel>/`.

- All figures
    - [done] merge the per-panel source data (`figures/8_source_data/*/*.xlsx`) into one file, one sheet per figure: `figures/8_source_data/Source_Data.xlsx` (`src/pipes/source_data/merge_source_data.py`; rerun it after any panel changes). It is 64 MB, above the journal's limit of about 30 MB per file: deposit it in a repository (e.g. Zenodo) or reduce the largest sheets (Supplementary Fig. 2, Fig. 7, Fig. 2, Fig. 3)
    - add "Source data are provided as a Source Data file." to every legend with source data, and "Source data are provided with this paper." to Data Availability
    - [done for all panels] Nature artwork guide (Arial, 5-7 pt, editable text, images of at least 300 dpi, .pdf next to each .svg): `src/nodes/figure_style.py`, applied in every panel notebook; the template panels are replotted from their source data with `src/pipes/figures/replot_from_source_data.py`. `fig3h/fig3h_clusters.svg` is an old duplicate of `fig3h/fig3h.svg` (same clusters panel), to be removed
    - main figures: one vector file per figure with editable text (.ai, .eps or .pdf), all panels labelled, no legend in the file
    - supplementary figures: all in one Supplementary Information PDF, each with a title and a legend of at most 350 words
    - bar graphs with one value per bar (Fig. 5a, 5d, 6b; Supplementary Figs. 1c, 5b, 6d, 6e, 9): state in the legend what one bar is (e.g. "each bar is one sorting")
- Fig. 1 (`paper/figures/fig1.pdf`; legend and main text in `paper/sn-article-editor.tex`)
    - [done] figure: Arial only, 5-7 pt text (panel letters 8 pt bold), RGB, images of at least 370 ppi, 177.8 x 184.9 mm; "innervation", "270°"
    - [done] legend: m-types, e-types, symbols of panel c and layer colours defined; 298 words (under 300, so about 185 mm high); edits marked with \newtext/\deletetext
    - [done] Reimann et al. and Isbister et al. cited by their eLife versions (2026, CC BY 4.0); main text: "Fig. 1d, g" and units in roman type (line 110)
    - [done] no bar graph in Fig. 1
    - main text line 365: "eight different directions, spaced 45 degrees apart" contradicts the Fig. 1 legend, main text line 114 and the Fig. 7a legend (360 directions separated by 1 degree; simulation `8slc_80f_360r_50t_200ms`): decide which is correct
    - checklist reply to "suspected third party content is present in figure 1c": "Fig. 1c is an original drawing by the authors, and the column renderings in Fig. 1d-e were generated by the authors from the published model (Reimann et al., Isbister et al., eLife, 2026). The figure contains no third-party material."
    - optional, in Illustrator: credit line "Reimann et al., 2026; Isbister et al., 2026" (now "Reimann et al, 2024; Isbister et al, 2024"); "φ" (10 pt) and the dots (11-12 pt) to 7 pt or less
    - optional: source data for the thalamocortical innervation profile of panel d
- Fig. 2
    - 2q, 2u: update figure and text
    - 2m-p, 2r-u: give the n (precise value), define the box plots (centre, box bounds, whiskers, outliers), give exact P values
- Fig. 3
    - 3a, 3b: create the source data (notebooks are empty); "check reply natcomm"
    - 3a, 3b: third-party panels. 3a is reproduced from Paulk et al. (bioRxiv 2021, CC BY-ND: copying allowed with credit, adaptation not; the Nature Neuroscience 2022 version is not open access). 3b is copied from Sun et al. (J. Physiol. 2021, Wiley, not open access). Get permission (RightsLink) or redraw from the authors' data; legend and main text cite different versions of Paulk et al. for 3a
    - 3f: update figure and text
- Fig. 4
    - 4a: update figure (one more L2/3 dot)
    - 4c: update figure and text (different seed)
    - 4b: give the n; 4b-c: define the box plots, state one- or two-sided tests, give exact P values
- Fig. 5
    - 5b: individual data points overlaid (done); legend: "medians (bars)" -> "means (bars)", dots are individual layers, the yield of 0 of E in L1 is not shown on the logarithmic scale; give the n
    - 5e, 5f, 5g: update figures and text (10-minute correction of the recording, see CLAUDE.md)
    - 5a, 5g: state one- or two-sided tests, give exact P values
- Fig. 6
    - 6c, 6d: update R-squared and the model (see CLAUDE.md); 6e, 6f: update weights
    - 6b: state whether the chi-square test is one- or two-sided
    - 6e-f: name the statistical test (two-sided Wald test) and give exact P values
- Fig. 7
    - 7h: give the n
    - 7b, 7f-j: define the box plots
- Supplementary Fig. 4
    - Methods or legend: the Kilosort3 evoked sorting is the one on DANDI (its counts and marks differ from the paper's run)
- Supplementary Fig. 5
    - 5a: individual data points overlaid (done); legend: "median yield (bar)" -> "mean yield (bar)", dots are individual layers, the yields of 0 (KS3 M, KS3 NS, KS4 E) are not shown on the logarithmic scale
- Supplementary Fig. 6
    - 6a: figure marker at 1.3 ms (done); main text, Methods and legend: 1.3 ms gives 96% of the maximum accuracy of Kilosort4 (maximum at 1.8 ms), not the maximum
    - 6b: legend: no peak of correctly detected units, a tail of intermediate scores (up to 0.4)
- Supplementary Fig. 9
    - the Kilosort 2.5 / S panel shows the Kilosort 3 numbers (bug in the paper's notebook): the V100 bar is 413 single-units / 1,162 multi-units; re-run Kilosort 2.5 on S on the RTX 5090 (with and without drift correction) or remove those two bars
    - Data Availability: the 18 RTX 5090 sortings (with and without drift correction) were not deposited