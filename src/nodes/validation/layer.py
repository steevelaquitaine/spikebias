"""Node to add metadata

The atlas is available open source here: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/QREN2T
The voxel package too

Returns:
    _type_: _description_
"""
from voxcell.nexus.voxelbrain import Atlas
import numpy as np
import os
from src.nodes import utils


def getAtlasInfo(data_conf, BlueConfig, electrodePositions: np.array):
    """_summary_

    Args:
        data_conf (_type_): _description_
        BlueConfig (_type_): _description_
        electrodePositions (np.array): N electrode x 3 dimensions

    Returns:
        _type_: _description_
    """

    # save path
    save_path = data_conf["metadata"]["atlas"]
    
    bluefile = open(BlueConfig,'r')
    bluelines = bluefile.readlines()
    bluefile.close()
    for line in bluelines:
        if 'Atlas' in line:
            atlasName = line.split('Atlas ')[-1].split('\n')[0]
            break
        
    atlas = Atlas.open(atlasName)
    brain_regions = atlas.load_data('brain_regions')
    region_map = atlas.load_region_map()
    regionList = []
    layerList = []

    for position in electrodePositions:
        try:
            for id_ in brain_regions.lookup([position]):
                region = region_map.get(id_, 'acronym')
                regionList.append(region.split(';')[0])
                layerList.append(region.split(';')[1])
        except:
            regionList.append('Outside')
            layerList.append('Outside')            
    
    # write
    utils.create_if_not_exists(save_path)
    np.save(os.path.join(save_path, "regions.npy"), regionList)
    np.save(os.path.join(save_path, "layers.npy"), layerList)
    return regionList, layerList


def loadAtlasInfo(data_conf: dict):

    # read path
    read_path = data_conf["metadata"]["atlas"]
    regions = np.load(os.path.join(read_path, "regions.npy"), allow_pickle=False).tolist()
    layers = np.load(os.path.join(read_path, "layers.npy"), allow_pickle=False).tolist()
    return regions, layers