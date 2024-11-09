from voxcell.nexus.voxelbrain import Atlas

def getAtlasInfo(BlueConfig, electrodePositions):

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
    return regionList, layerList
