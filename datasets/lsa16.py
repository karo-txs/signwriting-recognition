from dataclasses import dataclass
from dataset import Dataset

@dataclass
class LSA16(Dataset):
    
    name: str = "LSA16"
    
    def download(self):
        self.download_from_url("https://mega.nz/#!hcgQUCAT!WtTSPTO6GuXIXs1BuRJDGburN4FRHBAs9EzRfbd2ra4")
        return self
    
    def get_mapper(self):
        return {
            "1_": "S14c",
            "2_": "S144",
            "3_": "S1a0",
            "4_": "S16d",
            "5_": "S177",
            "6_": "S115",
            "7_": "S106",
            "8_": "S100",
            "9_": "S1dc",
            "10_": "S15a",
            "11_": "S15d",
            "12_": "S177",
            "13_": "S1f5",
            "14_": "S1f8",
            "15_": "S19a",
            "16_": "S10e",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/lsa16_raw/lsa16_raw',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        