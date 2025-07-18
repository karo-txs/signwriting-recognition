from interface.dataset import Dataset
from dataclasses import dataclass

@dataclass
class Hagrid(Dataset):
    
    name: str = "Hagrid"
    
    def download(self):
        self.download_from_kaggle("innominate817/hagrid-classification-512p")
        return self
    
    def get_mapper(self):
        return {
            "rock": "S1a0",
            "ok": "S1ce",
            "dislike": "S1f5",
            "like": "S1f5",
            "peace": "S10e",
            "peace_inverted": "S10e",
            "three2": "S11e",
            "palm": "S14c",
            "stop": "S15a",
            "stop_inverted": "S15a",
            "call": "S19a",
            "mute": "S100",
            "one": "S100",
            "two_up": "S115",
            "two_up_inverted": "S115",
            "four": "S144",
            "three": "S186",
            "fist": "S203",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/hagrid-classification-512p/hagrid-classification-512p', 
                                                target_dir=f'{self.get_base_path()}/{self.name}/sw-classified/train')
        