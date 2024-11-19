
# Analyse the simulated recordings from the dense probe at depth 1

author: laquitainesteeve@gmail.com

1. Download NWB file

Download the NWB file of the dense probe at depth 1 saved in DANDI archive.

```bash
sh cluster/dandi/download/raw/dense_spont_p1.sh
```

2. Preprocess

Fit the gain and missing noise, (re-)wire probe, high pass filter, common median reference the recording and write ground truth extractor separately

```bash
sh cluster/prepro/dense_spont/process_probe1_nwb.sh
```

3. Spike sort

Sort the wired recording with kilosort 4

```bash
sh cluster/sorting/dense_spont_from_nwb/probe_1/10m/ks4.sh
```