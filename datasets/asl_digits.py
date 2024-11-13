from dataclasses import dataclass
from dataset import Dataset

@dataclass
class ASLDigits(Dataset):
    
    name: str = "ASLDigits"
    
    def download(self):
        self.download_from_url("https://github.com/ardamavi/Sign-Language-Digits-Dataset/archive/refs/heads/master.zip")
        return self
    
    def get_mapper(self):
        return {
            "0": "S176",
            "1": "S100",
            "2": "S10e",
            "3": "S11e",
            "4": "S144",
            "5": "S14c",
            "6": "S186",
            "7": "S1a4",
            "8": "S1bb",
            "9": "S1ce",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/Sign-Language-Digits-Dataset-master/Dataset',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        