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
    - [done] main text line 365 (marked): the simulation has 360 directions (line 114, Fig. 1 and Fig. 7a legends, simulation `8slc_80f_360r_50t_200ms`); the information metrics (Fig. 7b-j) are computed for eight of them, 45 degrees apart (Methods); Results and Fig. 7 legend now say so. Not verifiable from the repository (the capacity code needs `mftma`)
    - checklist reply to "suspected third party content is present in figure 1c": "Fig. 1c is an original drawing by the authors, and the column renderings in Fig. 1d-e were generated by the authors from the published model (Reimann et al., Isbister et al., eLife, 2026). The figure contains no third-party material."
    - optional, in Illustrator: credit line "Reimann et al., 2026; Isbister et al., 2026" (now "Reimann et al, 2024; Isbister et al, 2024"); "φ" (10 pt) and the dots (11-12 pt) to 7 pt or less
    - optional: source data for the thalamocortical innervation profile of panel d
- Fig. 2
    - [done] 2q, 2t, 2u: panels redrawn from the source data (q: `anr_plots.py`; t, u: `ap_power_plots.py`; f: `single_site_plots.py`); main text: statistics of r-u across layers and across experiments updated, "largely" added to the ANR sentence; legend: n per box, box-plot definition, two-sided Kruskal-Wallis with Dunn's post hoc test and Holm-Sidak correction, "Source data are provided as a Source Data file." (all marked with \newtext/\deletetext); Methods: paragraph "Peak amplitude and background noise", statistical tests (Dunn's test, scikit-posthocs v0.9.0), Mann-Whitney U limited to the firing rates (Fig. 4)
    - [done] exact P values of 2m-p, 2r-u (editor: "in the legend, when possible"): the legend gives the overall Kruskal-Wallis result across layers (chi-square(4) = 33.6-232.1, P <= 8.8e-7) and states that all test statistics and exact P values are in the Source Data file (`fig2<panel>_statistics.xlsx`, written by `src/pipes/source_data/add_statistics.py`, 16 tables in the Fig. 2 sheet of `Source_Data.xlsx`; rerun the script and `merge_source_data.py` after any change of the panels). More than 60 pairwise comparisons do not fit in the legend: say so in the reply to the editor
    - figure `paper/figures/fig2.pdf`: panel k must show 27 traces at depth 1 (`fig2k_depth_1`; the placed panel has 24, the title says n=71); set the 14 characters below 5 pt to 5 pt (the "10" and "20" tick labels of the time axes in g-l, and the "2" of µV² in the y-labels of r and s); replace the non-Arial fonts (Book Antiqua µ, Myriad Pro, Footlight, DIN Condensed, Apple Braille) by Arial
    - panel f: the spikes of the paper's panel (about 210, 379, 725, 893 ms) are not in the source data (58, 388, 829 ms; site 10 of layer 5, 10th second of `recording_dense_probe2`): no site/second of that recording reproduces the paper's trace
    - dense-probe AP power (2t): the notebook uses 300-1,000 Hz (Neuropixels 2r: 300-2,000 Hz) while the main text and the legend say 0.3-2 kHz: decide which one is right (recompute 2t with 2,000 Hz, or correct the text); then add the AP power paragraph (and the fitting band 100-10,000 Hz of the slope) to the Methods
    - legend sentence "s and u were computed over 10 min": m-p also use the first 10 minutes (notebooks), r uses the entire traces, t 2,400 s: correct the sentence
    - 2u: the source data (`fig2u.xlsx`) differs from the paper's labels (newer run): Horvath L2/3 1.4 ± 0.05 -> 1.3 ± 0.06, L5 1.5 ± 0.05 -> 1.4 ± 0.05, L6 1.3 ± 0.05 -> 1.2 ± 0.05; dense-probe model L1 0.08 ± 0.01 -> 0.07 ± 0.02, L2/3 0.1 ± 0.01 -> 0.1 ± 0.02, L6 0.3 ± 0.06 -> 0.3 ± 0.05 (stars unchanged). Replot with statistics and labels done (`plot_slope_dense` in `src/nodes/validation/ap_power_plots.py`); check the main text and the Fig. 2 legend for quoted slope values
    - 2t: the source data (`figures/8_source_data/fig2t/fig2t.xlsx`, from `fig2t_new_stats.ipynb`) differs from the paper's panel (old computation, `fig2t_old_stats.ipynb`): Horvath L1 0.01 ± 7e-4 -> 0.02 ± 4e-4, L2/3 0.05 ± 0.01 -> 0.04 ± 0.01, L6 0.06 ± 0.01 -> 0.04 ± 8e-3; dense-probe model L6 9e-3 ± 5e-3 -> 8e-3 ± 5e-3 (the stars, all ***, are unchanged). Replot with statistics and labels done (`src/nodes/validation/ap_power_plots.py`). Still to do: check the main text (any AP power values of the dense probe) and the Fig. 2 legend, place the panel in the figure. **When the panel is final and the main text and the figure legend are updated, delete `notebooks/2_results_editorial_requests/old/fig2t_old_stats.ipynb` (and `old/fig2r_old_stats.ipynb`)** (its code is commented out; it only holds the old outputs)
- Fig. 3
    - [done] the former panels a (human spikes, Paulk et al., CC BY-ND) and b (cat spikes, Sun et al., Wiley) are removed (published images; the studies are cited in the main text); the remaining panels are renamed: former c-h are now a-f (`figures/8_source_data/fig3a`-`fig3f`, `notebooks/2_results_editorial_requests/fig3a.ipynb`-`fig3f.ipynb`; the two stubs are in `notebooks/2_results_editorial_requests/old/`). Checklist reply: "To avoid reusing previously published images, we removed the two panels reproducing published spike waveforms (human, Paulk et al.; cat, Sun et al.); these studies are cited in the main text."
    - [done] manuscript (`paper/sn-article-editor.tex`, marked \newtext / \deletetext): the legend and the main text use the new letters a-f; the Paulk and Sun et al. panels and the CC BY-ND credit are removed and the two studies are cited in the text; "Amplitudes in d" and "colored as in c ... in g and h" inside `paper/figures/fig3.pdf` must still be changed to b, and a, e and f, panel c must be replaced with `figures/8_source_data/fig3c/fig3c.svg` (row ids from the top), and the artboard must be rescaled to 180 mm wide with titles at 7 pt or less
    - 3d (former 3f, traces summed over all cells): update figure and text (checklist; what to update: to be defined)
    - `figures/8_source_data/fig3f/fig3f_clusters.svg` is an old duplicate of the clusters panel in `fig3f.svg`: to remove
- Fig. 4
    - 4a: update figure (one more L2/3 dot)
    - 4c: update figure and text (different seed)
    - [done, to decide] 4a, 4b, 4c recomputed with the evoked simulation sampled at 40 kHz (`dataset/01_intermediate/sorting/npx_evoked_40khz/SortingKS4_10m`, 273 single units) instead of the 20 kHz 1-hour simulation (407 single units): the median rate of the 20 kHz sorting (0.66 spikes/s in L5+L6, 77% below in vivo) is lower than the spontaneous one although the true rates are higher, because of 253 poorly matched low-rate units; at 40 kHz it is 1.46 spikes/s (-49%). M, NS and S are unchanged. Backup of the 20 kHz figures and notebooks: scratchpad `fig4_20khz/`. `firing_rate.get_sorted_unit_meta` now computes the rates as spike count / 600 s. Panels redrawn with `src/nodes/validation/firing_rate_plots.py` (4b, 4c); 4a from `fig4a.ipynb`. Not done: Supplementary Fig. 4 (still uses the 20 kHz evoked sortings; Figs. 5 and 7 use the 20 kHz evoked simulation and Fig. 6a-b the 10-minute 40 kHz one, see the legends); the Marques-Smith recording is 20.3 min (1,215 s), not 26 min: Methods corrected; place the new panels in `paper/figures/fig4.pdf`, update the legend, Results and Methods
    - 4b: give the n; 4b-c: define the box plots, state one- or two-sided tests, give exact P values
    - Marques-Smith: `dataset/01_intermediate/sorting/marques/SortingKS4_10m_rtx5090` is the first 10 minutes (0-600 s, 412 units, 855,581 spikes, properties kept) of the Kilosort 4 sorting of the whole recording (`SortingKS4_full_check`, `src/pipes/sorting/npx_vivo/full/ks4_duration_check.py`); median rate of the 97 good L5+L6 units: 1.98 spikes/s (the 10-minute sorting `SortingKS4_10m`: 2.82). The spontaneous simulation has a whole-length sorting on DANDI only (`sub-001-fitted/sub-001-fitted_ses-ks4.nwb`, first 10 minutes: 0.77 spikes/s for the 214 good L5+L6 units)
- Fig. 5
    - 5b: individual data points overlaid (done); legend: "medians (bars)" -> "means (bars)", dots are individual layers, the yield of 0 of E in L1 is not shown on the logarithmic scale; give the n
    - [done] Fig. 5 legend rewritten under 300 words (marked): means (bars), n, test of 5a (chi-square goodness-of-fit on the percentages of single-units, as in the figure; exact p in `fig5a_statistics.xlsx`), 100 random splits, two-sided Wald test of 5g, Source Data sentence; the Results about 5e-g and the 5a test (see `fig5a_statistics.xlsx`) are still to review
    - 5e, 5f, 5g: update figures and text (10-minute correction of the recording, see CLAUDE.md)
    - [done] 5e-g Results and Methods (marked): precision 88% and 93%, recall 62% and 54%; weights (mad_ratio, firing range, firing rate, Wald test); 75/25 split and first 10 minutes; log-transform values; the CEBRA comparison (Supplementary Fig. 5c-e) and the Lasso wording are not changed
    - [done] 5d redrawn from `fig5d.xlsx` (Kilosort4) with the geometry of panel d of the paper's figure, all text Arial 5 pt, editable (`src/nodes/validation/assignment_bias_plots.py`; same counts as the paper, bars and axes at the paper's positions to 0.1 pt; the panel letter is not drawn)
    - [done] 5e, 5f, 5g redrawn from the (10-minute) source data with the geometry of panels e, f, g of the paper's figure, all text Arial 5 pt (`src/nodes/models/Flc/curation_plots.py`; axes, ticks, dots, markers, bars and stars at the paper's positions to 0.1 pt; the y positions of the dots follow the new data; in g the tick values are -10, 0, 10 because the new weights are smaller, the axes keep the paper's size); the figure titles above e-f and the panel letters are not drawn; text of the Results about 5e-g is still to update
    - 5a, 5g: state one- or two-sided tests, give exact P values
- Fig. 6
    - [done] 6g redrawn from `fig6g.xlsx` with the geometry of panel g of the paper's figure (page 19), text Arial 5 pt (EXC and INH too) and title 7 pt, editable (`plot_neuron_type_distribution` in `src/nodes/validation/sorting_accuracy_plots.py`; the 35 rows, axes, line and text anchors at the paper's positions); canvas 129.12 x 409.6 pt = the paper's bounding box of the panel, y from 233.8 to 643.4 pt of the page (`FIG6GH_Y`), the same vertical range and origin as panel h, so that the rows of g and h align when placed at the same y; panel letter not drawn
    - [done] 6h redrawn from `fig6h.xlsx` in the paper's geometry (`plot_selection_biases`, same file): canvas 110.06 x 409.6 pt, the same y range (233.8-643.4 pt) and the same 35 row positions (y = 300.28 + i x 9.0545 pt) as 6g, so g and h line up without scaling; 5 pt Arial numbers (bold red/blue when p < 0.05), 7 pt title, editable. Note: the 6g in `paper/figures/fig6.pdf` was scaled by 1.0248 (row pitch 9.279 pt) to fit the previous 6h, which was not in the paper's geometry; place the new g and h unscaled
    - [TODO] 6h: correct the 210 Fisher exact tests (35 neuron types x 6 sorters) for multiple comparisons (Holm or Benjamini-Hochberg, in `fig6h.ipynb`; add the adjusted p-value to `fig6h.xlsx`), then update the colours of the panel, the Results sentence on over-represented interneurons (2.9 to 8.1 times), the legend (now "without correction for multiple comparisons") and the Methods. Check with the current p-values: 25 of 210 are below 0.05 uncorrected (about 10 are expected by chance); 13 stay below 0.05 with Bonferroni or Holm and 18 with Benjamini-Hochberg. Of the 20 significant pyramidal-cell (cADpyr) biases, 18 remain significant after Benjamini-Hochberg (the exceptions are L4 for ks3 and ks2.5), but none of the five rare-interneuron cells (ks2.5 L4 cACint 6.7, ks2.5 L5 cACint 2.9, ks2 L1 cACint 8.1, ks L5 bSTUT 4.1, ks L6 bSTUT 0.0) remains significant
    - [done] 6f redrawn from `fig6f.xlsx` (sheet weights) with the geometry of panel f of the paper's figure (page 19), text Arial 5 pt and title 7 pt, editable (`plot_feature_weights(df, "f", order)`; the rows follow the features of panel e, read from `fig6e.xlsx`; axes, bars and text anchors at the paper's positions; panel letter not drawn)
    - [done] 6e redrawn from `fig6e.xlsx` with the geometry of panel e of the paper's figure (page 19), text Arial 5 pt and titles 7 pt, editable (`plot_feature_weights(df, "e")` in `src/nodes/validation/sorting_accuracy_plots.py`; axes, ticks, bars and text anchors at the paper's positions; the features are sorted by weight, so their order differs from the paper's where the weights differ; panel letter not drawn)
    - [done] 6d redrawn from `fig6d.xlsx` and `fig6d_r2.xlsx` with the geometry of panel d of the paper's figure (page 19), text Arial 5 pt and title 7 pt, editable (`plot_model_accuracy(df, df_r2, "d")`; axes, ticks and R-squared 0.34 as in the paper); the panel letter is not drawn
    - [done] 6c redrawn from `fig6c.xlsx` and `fig6c_r2.xlsx` with the geometry of panel c of the paper's figure (page 19), text Arial 5 pt and title 7 pt, editable (`plot_model_accuracy` in `src/nodes/validation/sorting_accuracy_plots.py`); the R-squared is that of `fig6c_r2.xlsx` (0.41; the paper prints 0.43); the model update of CLAUDE.md (0.497) is not in the source data yet; the panel letter is not drawn
    - [done] 6b redrawn from `fig6b.xlsx` with the geometry of panel b of the paper's figure (page 19), text Arial 5 pt and titles 7 pt, editable (`plot_global_accuracy` in `src/nodes/validation/sorting_accuracy_plots.py`; the 12 bars and the axes are at the paper's positions to 0.1 pt, the counts and the p values are the paper's); the panel letter is not drawn
    - [done] 6a redrawn from `fig6a.xlsx` with the geometry of panel a of the paper's figure (page 19), text Arial 5 pt and titles 7 pt, editable (`src/nodes/validation/sorting_accuracy_plots.py`; axes, ticks, curves and dashed line at the paper's positions to 0.1 pt; the proportions of neurons above 0.8 are recomputed from the data and differ by 0.01 from the paper's for ks4, ks3, ks2.5 and ks2 in the synthetic model); the panel letter is not drawn
    - 6c, 6d: update R-squared and the model (see CLAUDE.md); 6e, 6f: update weights
    - 6b: state whether the chi-square test is one- or two-sided
    - 6e-f: name the statistical test (two-sided Wald test) and give exact P values
    - Still open (Fig. 6):
        - [TODO] 6h: multiple-comparison correction (see the TODO above); if applied, update the panel colours, the Results sentence on rare interneurons, the Discussion, the legend and the Methods
        - [TODO] Methods, Statistical tests: the Fisher test is described as a custom Monte Carlo test, while `fig6h.ipynb` uses the exact `scipy.stats.fisher_exact` (two-sided, one 2 x 2 table per neuron type and sorter)
        - [TODO] Methods, Model of sorting accuracy: add the cross-validation (100 random 75% / 25% splits, median McFadden R-squared over the splits that fit: 74 of 100 spontaneous, 99 of 100 evoked)
        - [TODO] legend 6c-d: R-squared is the cross-validated McFadden pseudo R-squared (median over 100 random splits), not the "explained variance"
        - [TODO] legend 6b: say what one bar represents (the pooled proportion across simulations for one spike sorter)
        - [TODO] legend: typos (missing full stops after "Kilosort4" and "each model neuron type", "Number annotations indicates", stray `}.}.` at the end)
        - [TODO] `paper/figures/fig6.pdf`: stray black tick at the right edge; EXC and INH labels close to panels d and f (the paper puts them just left of the panel g labels); check in Illustrator that all text is 5 pt (7 pt titles, 8 pt panel letters)
        - [TODO] checklist (.docx): Fig. 6 entries (source data list, n, statistical tests, vector figure); merge the Fig. 6 source data into `Source_Data.xlsx` (64 MB, above the ~30 MB limit)
        - [to decide] 6c, 6d show the paper's model (R-squared 0.41 and 0.34); the improved model (0.497 and 0.425, CLAUDE.md) is not in the figure or the text
- Fig. 7
    - [done] 7a redrawn from `fig7a.xlsx` with the geometry of panel a of the paper's figure (page 22): canvas 488.5 x 160.5 pt (bounding box of the panel, letter not drawn), each 2D and 3D manifold fitted (equal scales) in the rectangle of the paper's image, latent axes, titles and colour bar at the paper's positions, text Arial 5 pt and figure title 7 pt, editable (`src/nodes/validation/manifold_plots.py`; the 3D manifolds are images of 600 dpi, as in the paper). The point clouds are the source data's PCA (their orientation and shape differ a little from the paper's). Note: the paper's text in this panel is about 6 pt
    - [done] 7b redrawn from `fig7b.xlsx` with the geometry of panel b of the paper's figure (page 22): canvas 160.5 x 134.5 pt (panel with its title, letter not drawn), y axis 153.2 pt per unit of capacity / lower bound (1.0 at y = 371.71 pt), eight notched box plots 11.76 pt apart and 5.76 pt wide, red dashed mean of the ground truth, blue dashed chance level, labels at the paper's positions, text Arial 5 pt and title 7 pt, editable (`plot_information_capacity` in `src/nodes/validation/capacity_plots.py`). The title is on two lines: at 7 pt the paper's one-line title (148 pt wide at 6 pt) would be wider than the panel
    - [done] 7c redrawn from `fig7c.xlsx` with the geometry of panel c of the paper's figure (page 22): canvas 48.8 x 67.8 pt (panel letter not drawn; the bold title shared by panels c-f is not part of it), 3D manifold of the ground truth fitted in the rectangle of the paper's image, latent axes and title at the paper's positions, text Arial 5 pt, editable (`plot_single_manifold` in `src/nodes/validation/manifold_plots.py`, also usable for 7d and 7e)
    - [done] 7d redrawn from `fig7d.xlsx` with the geometry of panel d of the paper's figure (page 22): canvas 52.5 x 76.9 pt (panel letter not drawn), 3D manifold fitted in the rectangle of the paper's image, latent axes and three-line title at the paper's positions, text Arial 5 pt, editable (`plot_single_manifold(df, UNDERSAMPLED_PANEL)`). The paper's text is about 6 pt
    - [done] 7e redrawn from `fig7e.xlsx` with the geometry of panel e of the paper's figure (page 22): canvas 59.6 x 75.8 pt (panel letter not drawn), 3D manifold fitted in the rectangle of the paper's image, latent axes and four-line title at the paper's positions, text Arial 5 pt, editable (`plot_single_manifold(df, SELECTED_PANEL)`). The paper's text is about 6 pt
    - [done] 7f redrawn from `fig7f.xlsx` with the geometry of panel f of the paper's figure (page 22): canvas 116.8 x 120.5 pt (panel letter not drawn; the bold title shared by panels c-f is not part of it), y axis 105.35 pt per unit of capacity / lower bound (1.0 at y = 372.88 pt), 13 notched box plots (ground truth in black, undersampling in red and selection bias in white for each sorter; 3 pt wide, grey edges), red and blue dashed lines, legend and labels at the paper's positions, text Arial 5 pt, editable (`plot_sampling_capacity` in `src/nodes/validation/capacity_plots.py`). The paper's text is about 6 pt
    - [done] 7g redrawn from `fig7g.xlsx` with the geometry of panel g of the paper's figure (page 22): canvas 124.6 x 117.0 pt (panel letter not drawn), y axis 47.34 pt per unit of capacity / lower bound (1.0 at y = 514.65 pt), 13 notched box plots (ground truth in black, 'good + merged + partial' in red and degraded units in white for each sorter, 1.95 pt wide, grey edges, hollow fliers), red and blue dashed lines, legend and labels at the paper's positions, text Arial 5 pt, editable (`plot_quality_capacity` in `src/nodes/validation/capacity_plots.py`). The paper's text is about 6 pt
    - [done] 7h redrawn from `fig7h.xlsx` with the geometry of panel h of the paper's figure (page 22): canvas 110.25 x 92.57 pt (panel letter not drawn; the bold title shared by panels h-j is not part of it), y axis 232.9 pt per unit of radius (0.8 at y = 500.01 pt), eight notched black box plots (5.1 pt wide), red dashed mean of the ground truth, labels at the paper's positions, text Arial 5 pt, editable (`plot_radius` in `src/nodes/validation/capacity_plots.py`). The paper's text is about 6 pt
    - [done] 7i redrawn from `fig7i.xlsx` with the geometry of panel i of the paper's figure (page 22): canvas 108.6 x 114.0 pt (panel letter not drawn; the bold title shared by panels h-j is not part of it), y axis 7.13 pt per unit of dimension (14 at y = 487.41 pt), eight notched black box plots (5.1 pt wide), red dashed mean of the ground truth, title '200 bootstraps' and labels at the paper's positions, text Arial 5 pt, editable (`plot_dimension` in `src/nodes/validation/capacity_plots.py`, same code as 7h). The paper's text is about 6 pt
    - [done] 7j redrawn from `fig7j.xlsx` with the geometry of panel j of the paper's figure (page 22): canvas 113.6 x 92.6 pt (panel letter not drawn; the bold title shared by panels h-j is not part of it), y axis 1,449 pt per unit of correlation (0.14 at y = 517.17 pt), eight notched black box plots (5.1 pt wide), red dashed mean of the ground truth, labels at the paper's positions, text Arial 5 pt, editable (`plot_correlation` in `src/nodes/validation/capacity_plots.py`)
    - [TODO] 7b, 7f, 7g, 7h, 7i, 7j: the red dashed line of the paper's panels is at the MEAN of the 200 ground truth bootstraps (e.g. 7j: 0.14153; the median is 0.14068), not at the median as the label of panel 7b ("Ground truth median capacity") says; the redrawn panels draw it at the mean and 7b is labelled "mean capacity": check that the Fig. 7 legend and text do not call it the median (they do not mention the line)
    - 7h: give the n
    - 7b, 7f-j: define the box plots
- Supplementary Fig. 2
    - [done] 2a redrawn from `supp_fig2a.xlsx` with the geometry of panel a of `paper/figures/supp_fig2.pdf`: canvas 120.1 x 191.1 pt (panel letter not drawn), axes, ticks (major at 10^1 and 10^3 Hz, minor at 0.2-0.8 of each decade), dots, power laws of 1-90 Hz and legend at the paper's positions, text Arial 5 pt, editable (`plot_psd` in `src/nodes/validation/psd_plots.py`). The y label reads "Normalized power (ratio)" (the paper's figure has the typo "Normalzed"); the legend keeps the entry "S: Synthetic" of the paper although this experiment has no points (see the notebook)
    - [done] 2b redrawn from `supp_fig2b.xlsx` with the geometry of panel b of `paper/figures/supp_fig2.pdf`: canvas 277.9 x 296.8 pt (panel letter not drawn), five layer panels (L1, L4 / L2/3, L5, L6) with their axes, ticks, power laws of 100-10,000 Hz, bold 7 pt titles and labels at the paper's positions, text Arial 5 pt, editable (`plot_psd_layers` in `src/nodes/validation/psd_plots.py`); y label "Normalized power (ratio)" (typo "Normalzed" in the paper's figure). The power laws of the xlsx differ slightly from those drawn in the paper's figure (up to about 1.7 pt, e.g. the Horvath line of L1 drops by 28.1 pt in the xlsx and 25.5 pt in the paper)
    - [done] 2c redrawn from `supp_fig2c.xlsx` with the geometry of panel c of `paper/figures/supp_fig2.pdf`: canvas 151.2 x 159.5 pt (panel letter not drawn), five traces (grey, 0.3 pt) spanning their minimum and maximum (the labels of the y axes), 4 s axis, coloured layer labels, bold 7 pt title (centred 2.5 pt left of the paper's, to stay inside the canvas), labels, text Arial 5 pt, editable (`plot_traces` in `src/nodes/validation/trace_plots.py`, also usable for 2d-g). The traces match the paper's figure
    - [done] 2d redrawn from `supp_fig2d.xlsx` with the geometry of panel d of `paper/figures/supp_fig2.pdf` (`SIM_NPX_SPONT` in `trace_plots.py`): height 152.1 pt as the paper's, width 140.0 pt instead of 130.1 pt (widened by the 7 pt bold title, which is wider than the paper's 6 pt one), y axes spanning the rounded minimum and maximum, text Arial 5 pt, editable. The spikes of L4 and L5 are at the same times and amplitudes as in the paper's figure but the noise differs (the y limits of the paper, e.g. L1 16/-14 uV, differ from those of the data, 12/-13 uV): the recording on disk is not the one of the paper's figure
    - [done] 2e redrawn from `supp_fig2e.xlsx` with the geometry of panel e of `paper/figures/supp_fig2.pdf` (`SIM_NPX_EVOKED` in `trace_plots.py`): height 153.0 pt as the paper's, width 131.7 pt instead of 129.4 pt (widened by the 7 pt bold title), y axes spanning the rounded minimum and maximum, text Arial 5 pt, editable. As in 2d, the spikes and the evoked responses are those of the paper's figure but the noise and the y limits differ (e.g. L1 17/-18 uV in the paper, 13/-14 uV in the data)
    - [done] 2f redrawn from `supp_fig2f.xlsx` with the geometry of panel f of `paper/figures/supp_fig2.pdf` (`IN_VIVO_DENSE` in `trace_plots.py`): canvas 145.9 x 153.0 pt (panel letter not drawn), five traces spanning their minimum and maximum (the labels of the y axes), coloured layer labels, bold 7 pt title, labels at the paper's positions, text Arial 5 pt, editable. The traces and the y limits match the paper's figure (in vivo recording); the traces are vector paths here, raster images in the paper
    - [done] 2g redrawn from `supp_fig2g.xlsx` with the geometry of panel g of `paper/figures/supp_fig2.pdf` (`SIM_DENSE_SPONT` in `trace_plots.py`): height 151.6 pt as the paper's, width 141.4 pt instead of 131.5 pt (widened by the 7 pt bold title), y axes spanning the rounded minimum and maximum, text Arial 5 pt, editable. The traces do not match the paper's figure: the y limits differ (e.g. L4 300/-566 uV in the paper, 199/-309 uV in the data) and the spikes of L4 and L5 are at other times (only those of L6 coincide), so the site or the recording of the paper's figure differs from the one on disk
- Supplementary Fig. 3
    - [done] 3a-f redrawn from their source data and rescaled: the paper's figure is 241 mm high (page of 510 x 684 pt), the new panels keep its arrangement with every length multiplied by 0.735 (`SCALE` in `src/nodes/validation/spike_shape_plots.py`), so that the stacked panels (a and b side by side, c, d-f side by side) are 181.4 mm high and 131 mm wide (125 mm for c; guideline: at most 185 mm for a legend of less than 300 words); text Arial 5 pt, titles 7 pt (bold; unit titles of c regular), editable. Code: `spike_shape_plots.py` (a, b; subplot positions measured on the paper's figure in `spike_shape_layout.py`) and `burst_plots.py` (c-f). Canvas sizes (pt): a 154 x 330, b 155 x 312, c 346 x 81, d 128 x 103, e 111 x 102, f 113 x 103; place them at the paper's relative positions
    - 3f: the y label of the paper's figure reads "Trough widths (uV)" but the widths are in ms: the new label reads "(ms)"
    - 3d-f: the box plots are drawn from the trough amplitudes and widths of the source data (notched boxes, whiskers at 1.5 IQR); the quartiles of the boxes of the paper's panel d differ from those computed from the xlsx by up to 3 uV (the medians, whisker ends and outliers agree): to check for e and f
    - 3c: n_bursts is 2, 2 and 62 for the three units: give it in the legend (the 95% CI of the first two units is over two bursts)
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