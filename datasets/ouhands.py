from dataclasses import dataclass
from dataset import Dataset

@dataclass
class OUHANDS(Dataset):
    
    name: str = "ouhands"
    
    def download(self):
        self.download_from_kaggle("mumuheu/ouhands")
        return self
    
    def get_mapper(self):
        return {
            "A": "S1f8",
            "B": "S14c",
            "C": "S100",
            "D": "S1dc",
            "E": "S10e",
            "F": "S186",
            "H": "S1a0",
            "I": "S192",
            "J": "S147",
            "K": "S115"
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/OUHANDS_test/test/hand_data/colour', 
                                                target_dir=f'{self.get_base_path()}/{self.name}_test/images')
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/OUHANDS_train/train/hand_data/colour', 
                                                target_dir=f'{self.get_base_path()}/{self.name}_train/images')