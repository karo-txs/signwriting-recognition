from dataclasses import dataclass
from dataset import Dataset

@dataclass
class BengaliAlphabet(Dataset):
    
    name: str = "bengali_alphabet"
    
    def download(self):
        self.download_from_kaggle("muntakimrafi/bengali-sign-language-dataset")
        return self
    
    def get_mapper(self):
        return {
            "0": "S16c",
            "1": "S17b",
            "2": "S107",
            "3": "S101",
            "4": "S11a",
            "5": "S175",
            "6": "S194",
            "7": "S177",
            "8": "S195",
            "9": "S16d",
            "10": "S169",
            "11": "S170",
            "12": "S1eb",
            "13": "S1d4",
            "14": "S1ce",
            "15": "S1a3",
            "16": "S150",
            "17": "S12d",
            "18": "S1ce",
            "19": "S1fb",
            "20": "S1f4",
            "21": "S124",
            "22": "S101",
            "23": "S201",
            "24": "S1dc",
            "25": "S1f5",
            "26" :"S1f9",
            "27": "S1b2", # Verify
            "28": "S142",
            "29": "S1cd",
            "30": "S147",
            "31": "S18c",
            "32": "S1dc",
            "33": "S14a",
            "34": "S115",
            "35": "S1e1",
            "36": "S1f4",
            "37": "S176"
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/RESIZED_DATASET', 
                                                target_dir=f'{self.get_base_path()}/{self.name}_train/images')
        
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/RESIZED_TESTING_DATA', 
                                                target_dir=f'{self.get_base_path()}/{self.name}_test/images')