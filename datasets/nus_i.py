from dataclasses import dataclass
from dataset import Dataset

@dataclass
class NUSI(Dataset):
    
    name: str = "nus_i"
    
    def download(self):
        self.download_from_url("https://www.ece.nus.edu.sg/stfpage/elepv/NUS-HandSet/NUS-Hand-Posture-Dataset-I.zip")
        return self
    
    def get_mapper(self):
        return {
            "g1": "S15a",
            "g2": "S15d",
            "g3": "S1d7",
            "g4": "S1da",
            "g5": "S100",
            "g6": "S115",
            "g7": "S1f8",
            "g8": "S1a3",
            "g9": "S177",
            "g10": "S182",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/NUS Hand Posture Dataset/Color',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        