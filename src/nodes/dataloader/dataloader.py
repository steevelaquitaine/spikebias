"""Module to load datasets

Returns:
    _type_: _description_
"""

from dandi.dandiapi import DandiAPIClient
import spikeinterface.extractors as se

class SortingLoader:
    """Load SortingExtractors from dandi archive
    """
    def __init__(self, dandiset_id, filepath, sfreq, tstart):
        self.dandiset_id = dandiset_id
        self.filepath = filepath
        self.sfreq = sfreq
        self.tstart = tstart
        self.sorting = None

    def load_sorting(self):
        with DandiAPIClient() as client:
            asset = client.get_dandiset(self.dandiset_id, 'draft').get_asset_by_path(self.filepath)
            s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
        self.sorting = se.NwbSortingExtractor(
            file_path=s3_path, 
            stream_mode="remfile", 
            use_pynwb=True,  
            sampling_frequency=self.sfreq, 
            t_start=self.tstart
        )
        return self.sorting

