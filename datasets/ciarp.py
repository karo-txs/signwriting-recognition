from dataclasses import dataclass
from dataset import Dataset

@dataclass
class CIARP(Dataset):
    
    name: str = "CIARP"
    
    def download(self):
        self.download_from_url("https://home.agh.edu.pl/~bkw/code/ciarp2017/ciarp.zip")
        return self
    
    def get_mapper(self):
        return {
            "g1": "15a10",
            "g2": "15d10",
            "g3": "1d710",
            "g4": "1da10",
            "g5": "10010",
            "g6": "11510",
            "g7": "1f810",
            "g8": "1a310",
            "g9": "17710",
            "g10": "18210",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/NUS Hand Posture Dataset/Color',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        