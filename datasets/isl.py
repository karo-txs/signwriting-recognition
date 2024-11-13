from dataclasses import dataclass
from dataset import Dataset

@dataclass
class ISL(Dataset):
    
    name: str = "ISL"
    
    def download(self):
        self.download_from_url("https://github.com/marlondcu/ISL/raw/master/Frames/Person1.zip", ignore_exists=True)
        self.download_from_url("https://github.com/marlondcu/ISL/raw/master/Frames/Person2.zip", ignore_exists=True)
        self.download_from_url("https://github.com/marlondcu/ISL/raw/master/Frames/Person3.zip", ignore_exists=True)
        self.download_from_url("https://github.com/marlondcu/ISL/raw/master/Frames/Person4.zip", ignore_exists=True)
        self.download_from_url("https://github.com/marlondcu/ISL/raw/master/Frames/Person5.zip", ignore_exists=True)
        self.download_from_url("https://github.com/marlondcu/ISL/raw/master/Frames/Person6.zip", ignore_exists=True)
        return self
    
    def get_mapper(self):
        return {
            "Person1-A": "S1f8",
            "Person2-A": "S1f8",
            "Person3-A": "S1f8",
            "Person4-A": "S1f8",
            "Person5-A": "S1f8",
            "Person6-A": "S1f8",
            
            "Person1-B": "S147",
            "Person2-B": "S147",
            "Person3-B": "S147",
            "Person4-B": "S147",
            "Person5-B": "S147",
            "Person6-B": "S147",
            
            "Person1-C": "S16d",
            "Person2-C": "S16d",
            "Person3-C": "S16d",
            "Person4-C": "S16d",
            "Person5-C": "S16d",
            "Person6-C": "S16d",
            
            "Person1-D": "S100",
            "Person2-D": "S100",
            "Person3-D": "S100",
            "Person4-D": "S100",
            "Person5-D": "S100",
            "Person6-D": "S100",
            
            "Person1-E": "S14a",
            "Person2-E": "S14a",
            "Person3-E": "S14a",
            "Person4-E": "S14a",
            "Person5-E": "S14a",
            "Person6-E": "S14a",
            
            "Person1-F": "S1d2",
            "Person2-F": "S1d2",
            "Person3-F": "S1d2",
            "Person4-F": "S1d2",
            "Person5-F": "S1d2",
            "Person6-F": "S1d2",
            
            "Person1-G": "S1ce",
            "Person2-G": "S1ce",
            "Person3-G": "S1ce",
            "Person4-G": "S1ce",
            "Person5-G": "S1ce",
            "Person6-G": "S1ce",
            
            "Person1-H": "S1a0",
            "Person2-H": "S1a0",
            "Person3-H": "S1a0",
            "Person4-H": "S1a0",
            "Person5-H": "S1a0",
            "Person6-H": "S1a0",
            
            "Person1-I": "S192",
            "Person2-I": "S192",
            "Person3-I": "S192",
            "Person4-I": "S192",
            "Person5-I": "S192",
            "Person6-I": "S192",
            
            "Person1-J": "S192",
            "Person2-J": "S192",
            "Person3-J": "S192",
            "Person4-J": "S192",
            "Person5-J": "S192",
            "Person6-J": "S192",
            
            "Person1-K": "S1ba",
            "Person2-K": "S1ba",
            "Person3-K": "S1ba",
            "Person4-K": "S1ba",
            "Person5-K": "S1ba",
            "Person6-K": "S1ba",
            
            "Person1-L": "S15d",
            "Person2-L": "S15d",
            "Person3-L": "S15d",
            "Person4-L": "S15d",
            "Person5-L": "S15d",
            "Person6-L": "S15d",
            
            "Person1-M": "S185",
            "Person2-M": "S185",
            "Person3-M": "S185",
            "Person4-M": "S185",
            "Person5-M": "S185",
            "Person6-M": "S185",
            
            "Person1-N": "S119",
            "Person2-N": "S119",
            "Person3-N": "S119",
            "Person4-N": "S119",
            "Person5-N": "S119",
            "Person6-N": "S119",
            
            "Person1-O": "S176",
            "Person2-O": "S176",
            "Person3-O": "S176",
            "Person4-O": "S176",
            "Person5-O": "S176",
            "Person6-O": "S176",
            
            "Person1-P": "S18c",
            "Person2-P": "S18c",
            "Person3-P": "S18c",
            "Person4-P": "S18c",
            "Person5-P": "S18c",
            "Person6-P": "S18c",
            
            "Person1-Q": "S1a4",
            "Person2-Q": "S1a4",
            "Person3-Q": "S1a4",
            "Person4-Q": "S1a4",
            "Person5-Q": "S1a4",
            "Person6-Q": "S1a4",
            
            "Person1-R": "S11a",
            "Person2-R": "S11a",
            "Person3-R": "S11a",
            "Person4-R": "S11a",
            "Person5-R": "S11a",
            "Person6-R": "S11a",
            
            "Person1-S": "S203",
            "Person2-S": "S203",
            "Person3-S": "S203",
            "Person4-S": "S203",
            "Person5-S": "S203",
            "Person6-S": "S203",
            
            "Person1-T": "S1ea",
            "Person2-T": "S1ea",
            "Person3-T": "S1ea",
            "Person4-T": "S1ea",
            "Person5-T": "S1ea",
            "Person6-T": "S1ea",
            
            "Person1-U": "S115",
            "Person2-U": "S115",
            "Person3-U": "S115",
            "Person4-U": "S115",
            "Person5-U": "S115",
            "Person6-U": "S115",
            
            "Person1-V": "S10e",
            "Person2-V": "S10e",
            "Person3-V": "S10e",
            "Person4-V": "S10e",
            "Person5-V": "S10e",
            "Person6-V": "S10e",
            
            "Person1-W": "S186",
            "Person2-W": "S186",
            "Person3-W": "S186",
            "Person4-W": "S186",
            "Person5-W": "S186",
            "Person6-W": "S186",
            
            "Person1-X": "S1ec",
            "Person2-X": "S1ec",
            "Person3-X": "S1ec",
            "Person4-X": "S1ec",
            "Person5-X": "S1ec",
            "Person6-X": "S1ec",
            
            "Person1-Y": "S19a",
            "Person2-Y": "S19a",
            "Person3-Y": "S19a",
            "Person4-Y": "S19a",
            "Person5-Y": "S19a",
            "Person6-Y": "S19a",
            
            "Person1-Z": "S10c",
            "Person2-Z": "S10c",
            "Person3-Z": "S10c",
            "Person4-Z": "S10c",
            "Person5-Z": "S10c",
            "Person6-Z": "S10c",
            
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        