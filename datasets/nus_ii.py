from dataclasses import dataclass
from dataset import Dataset

@dataclass
class NUSII(Dataset):
    
    name: str = "nus_ii"
    
    def download(self):
        self.download_from_url("https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/NUS-Hand-Posture-Dataset-II.zip")
        return self
    
    def get_mapper(self):
        return {
            "a": "S1f8",
            "b": "S15a",
            "c": "S115",
            "d": "S19a",
            "e": "S1ce",
            "f": "S182",
            "g": "S17d",
            "h": "S1dc",
            "i": "S177",
            "j": "S115",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/NUS-Hand-Posture-Dataset-II/NUS Hand Posture dataset-II/Hand Postures',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        