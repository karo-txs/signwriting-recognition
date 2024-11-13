from dataclasses import dataclass
from dataset import Dataset

@dataclass
class HG14(Dataset):
    
    name: str = "HG14"
    
    def download(self):
        self.download_from_kaggle("gulerosman/hg14-handgesture14-dataset")
        return self
    
    def get_mapper(self):
        return {
            "Gesture_0": "S203",
            "Gesture_1": "S100",
            "Gesture_2": "S10e",
            "Gesture_3": "S186",
            "Gesture_4": "S144",
            "Gesture_5": "S14c",
            "Gesture_6": "S19a",
            "Gesture_7": "S177",
            "Gesture_8": "S1f0",
            "Gesture_9": "S1f7",
            "Gesture_10": "S193",
            "Gesture_11": "S19c",
            "Gesture_12": "S1a0",
            "Gesture_13": "S15a",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/HG14/HG14-Hand Gesture', 
                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        