
# Setup coding environment

author: steeve.laquitaine@epfl.ch, laquitainesteeve@gmail.com

## Specs 

* Tested on a MacOS Catalina, v10.15.7 laptop.

## Prerequisites

1. Install Python 3.9.7
2. Install mamba 1.5.11 to create virtual environments (unlike pip, installs already compiled binaries)

## Create python virtual environments

* `env/spikebias.yml`: the main environment for most analyses
* `env/write_nwb.txt`: to convert RecordingExtractor and sortingExtractor to NWB files
* `env/dandi2.txt`: to upload to dandi archive
