from interface.dataset import Dataset
from dataclasses import dataclass


@dataclass
class LSWH100(Dataset):
    
    name: str = "LSWH100"
    
    def download(self):
        self.download_zenodo_record(10635628)
        return self
    
    def get_mapper(self):
        pass
    
    def map_classes(self):
        pass