from dataclasses import dataclass
from dataset import Dataset

@dataclass
class ISL2(Dataset):
    
    name: str = "ISL2"
    
    def download(self):
        self.download_from_url("https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/n34wm8sb3x-1.zip")
        return self
    
    def get_mapper(self):
        return {
            "A": "S1f8",
            "B": "S147",
            "C": "S17d",
            "D": "S101",
            "E": "S176",
            "F": "S1ce",
            "G": "S100",
            "H": "S115",
            "I": "S192",
            "J": "S192",
            "K": "S10e",
            "L": "S1dc",
            "M": "S201",
            "N": "S1fc",
            "O": "S176",
            "P": "S140",
            "Q": "S1f0",
            "R": "S11d",
            "S": "S203",
            "T": "S1ea",
            "U": "S115",
            "V": "S10e",
            "W": "S186",
            "X": "S10a",
            "Y": "S19a",
            "Z": "S100",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/ISL Hand Gesture Dataset/ISL custom Data/ISL custom Data',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        