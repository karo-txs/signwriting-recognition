from dataclasses import dataclass
from dataset import Dataset

@dataclass
class ASLAlphabet(Dataset):
    
    name: str = "asl_alphabet"
    
    def download(self):
        self.download_from_kaggle("grassknoted/asl-alphabet")
        return self
    
    def get_mapper(self):
        return {
            "F": "S1ce",
            "L": "S1dc",
            "T": "S1e1",
            "Q": "S1ed",
            "A": "S1f8",
            "X": "S10a",
            "K": "S10e",
            "V": "S10e",
            "R": "S11a",
            "E": "S14a",
            "space": "S16c",
            "C": "S16d",
            "M": "S18d",
            "Y": "S19a",
            "G": "S100",
            "Z": "S100",
            "D": "S101",
            "P": "S101",
            "H": "S115",
            "U": "S115",
            "N": "S119",
            "B": "S147",
            "O": "S176",
            "W": "S186",
            "I": "S192",
            "J": "S192",
            "S": "S203",
            "del": "S182"
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/asl_alphabet_train/asl_alphabet_train', 
                                                target_dir=f'{self.get_base_path()}/{self.name}_train/images')
        
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/asl_alphabet_test/asl_alphabet_test', 
                                                target_dir=f'{self.get_base_path()}/{self.name}_test/images')