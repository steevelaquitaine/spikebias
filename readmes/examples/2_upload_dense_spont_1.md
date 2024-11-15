
# Analyse the simulated recordings from the dense probe at depth 1

author: laquitainesteeve@gmail.com

1. Write NWB file

Write the NWB file of the dense probe recording and sorting extractors at depth 1 

```bash
sh cluster/dandi/write_nwb/raw/dense_spont_p1.sh
```

Write the NWB file of the dense probe recording and sorting extractors at depth 1 with noise and gain fitted to Horvath dataset

```bash
sh cluster/dandi/write_nwb/fitted/dense_spont_p1.sh
```

2. Upload NWB files updates to DANDI archive

```bash
sh cluster/dandi/upload/fitted/upload.sh
```

3. Download dense_spont_p1's NWB file from DANDI archive

```bash
sh cluster/dandi/download/fitted/dense_spont_p1.sh
```