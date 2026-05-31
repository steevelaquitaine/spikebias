# SPDX-License-Identifier: GPL-3.0-or-later
import sys
import pandas as pd
from bluerecording.writeH5 import writeH5File
from bluerecording.utils import process_writeH5_inputs

if __name__=='__main__':

    path_to_simconfig = '/home/joseph-tharayil/Documents/bluebrainStuff/csd_paper/electrodes/simulation_config.json'
    path_to_segment_positions = '/home/joseph-tharayil/Documents/bluebrainStuff/csd_paper/electrodes/positions'
    numNeuronsPerPositionFile = 1000
    numPositionFilesPerFolder = 50
    conductivity = [0.277] # in S/m
    circuitSubvolume = 'hex0'

    writeH5File(path_to_simconfig,path_to_segment_positions,'lfp_fullNeuropixels.h5',numNeuronsPerPositionFile,numPositionFilesPerFolder,conductivity,circuitSubvolume,None,None)

    writeH5File(path_to_simconfig,path_to_segment_positions,'lfp_fullNeuropixels_finite.h5',numNeuronsPerPositionFile,numPositionFilesPerFolder,conductivity,circuitSubvolume,None,None)
