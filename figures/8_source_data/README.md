# Source data

Source data of the figures of the paper, as requested by Nature Communications: one Excel file, `Source_Data.xlsx`, with one sheet per figure ("Fig. 2" to "Fig. 7", "Supplementary Fig. 1" to "Supplementary Fig. 9").

## Content

* `Source_Data.xlsx`: in each sheet, the panels follow each other from top to bottom. Each table starts with a title row in bold ("Fig. 2a"; the file and sheet names are added when a panel has several tables), then the column names and the data, followed by an empty row.
* `<panel>/` (e.g. `fig2a/`, `supp_fig6c/`): the source data of one panel (`.xlsx`) and its figure (`.svg`), written by the notebook of the same name in `notebooks/2_results_editorial_requests/` (e.g. `fig2a.ipynb`). Each notebook replots its panel from the saved `.xlsx`, which shows that the source data is sufficient to reproduce the panel.

Fig. 1 has no source data (it shows no data graphs).

## How to generate `Source_Data.xlsx`

1. Run the notebook of each panel in `notebooks/2_results_editorial_requests/` (kernel `spikebias`, environment `envs/spikebias.yml`). Each notebook writes `figures/8_source_data/<panel>/<panel>.xlsx` and `.svg`. The data the notebooks need are downloaded from DANDI (dandiset 001250) with `0_download_recordings.ipynb`, `0_download_ground_truths.ipynb` and `0_download_sortings.ipynb`.
2. Merge the panels into one file (about 1.5 minutes), from the root of the repository:

```bash
PYTHONPATH=$PWD envs/spikebias/bin/python src/pipes/source_data/merge_source_data.py
```

The script reads every `figures/8_source_data/<panel>/*.xlsx` and writes `figures/8_source_data/Source_Data.xlsx`. Rerun it whenever a panel's source data changes.

## Size

`Source_Data.xlsx` is about 64 MB. The largest sheets are Supplementary Fig. 2 (926,058 rows), Fig. 7 (214,491 rows), Fig. 2 (169,189 rows) and Fig. 3 (132,334 rows).
