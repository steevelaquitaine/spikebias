# SPDX-License-Identifier: GPL-3.0-or-later
import sys
from bluerecording.writeH5_prelim import initializeH5File

if __name__=='__main__':

    '''
    path_to_simconfig refers to the simulation_config from the 1-timestep simulation used to get the segment positions
    electrode_csv is a csv file containing the position, region, and layer of each electrode
    type is either LineSource or Reciprocity
    '''

    electrode_csvs = ['lfp_fullNeuropixels.csv','lfp_fullNeuropixels_finite.csv']

    path_to_simconfig = '/home/joseph-tharayil/Documents/bluebrainStuff/csd_paper/electrodes/simulation_config.json'

    outputfiles = ['lfp_fullNeuropixels.h5','lfp_fullNeuropixels_finite.h5']

    for i in range(2):
        initializeH5File(path_to_simconfig,outputfiles[i],electrode_csvs[i],'hex0')
