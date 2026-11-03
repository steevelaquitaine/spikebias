# Trained models

## cebra/

CEBRA models (`cebra==0.4.0`) that classify the sorted single-units of Kilosort4 into 'good' (agreement score of at least 0.8 with a ground truth neuron) and degraded units from their full spike waveforms and quality metrics (Supplementary Fig. 5c-e, notebooks `supp_fig5c.ipynb`, `supp_fig5d.ipynb`, `supp_fig5e.ipynb`).

| folder | regime | training iterations | used in |
|---|---|---|---|
| `spont_40_iterations/` | spontaneous | 40 | Supplementary Fig. 5c |
| `evoked_40_iterations/` | evoked | 40 | Supplementary Fig. 5d |
| `spont_100_iterations/` | spontaneous | 100 | Supplementary Fig. 5e |

Each folder holds 100 cross-validated models (one per random split of the units):

* `fold_000.pt` ... `fold_099.pt`: weights of the model of each fold (about 4.6 MB each); not tracked by git, published with the dataset
* `folds.csv`: for each fold, its precision and recall and the row indices of the units used for training and testing, their labels and the predictions
* `units.csv`: sorted unit id and label of each row of the dataset (the rows that `folds.csv` refers to)

Load a model (the files are torch checkpoints: only load files that you trust):

```python
from src.nodes.models.CebraSpike import utils
model = utils.load_crossval_model("models/cebra/spont_40_iterations", fold=0)
embedding = model.transform(data)       # data: units x (25 spikes x 240 samples + 11 quality metrics)
```

Requires `cebra==0.4.0` and `scikit-learn==1.5.2` (see `envs/cebraspike.txt`).
