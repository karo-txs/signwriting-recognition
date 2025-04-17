from interface.dataset import Dataset
from dataclasses import dataclass

@dataclass
class Hagrid(Dataset):
    
    name: str = "Hagrid"
    
    def download(self):
        self.download_from_kaggle("innominate817/hagrid-classification-512p")
        return self
    
    def get_mapper(self):
        return {
            "rock": "S1a0",
            "Gesture_1": "S1ce",
            "Gesture_2": "S1f5",
            "Gesture_3": "S10e",
            "Gesture_4": "S11e",
            "Gesture_5": "S14c",
            "Gesture_6": "S15a",
            "Gesture_7": "S19a",
            "Gesture_8": "S100",
            "Gesture_9": "S115",
            "Gesture_10": "S144",
            "Gesture_11": "S186",
            "Gesture_12": "S203",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/HG14/HG14-Hand Gesture', 
                                                target_dir=f'{self.get_base_path()}/{self.name}/sw-classified/test')
        