from dataclasses import dataclass
from dataset import Dataset

@dataclass
class ASLFingerSpelling(Dataset):
    
    name: str = "ASLFingerSpelling"
    
    def download(self):
        self.download_from_url("http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2")
        return self
    
    def get_mapper(self):
        return {
            "a": "S1f8",
            "b": "S15a",
            "c": "S172",
            "d": "S101",
            "e": "S14a",
            "f": "S1ce",
            "g": "S100",
            "h": "S115",
            "i": "S192",
            "k": "S140",
            "l": "S1dc",
            "m": "S1fd",
            "n": "S1fc",
            "o": "S176",
            "p": "S140",
            "q": "S1f4",
            "r": "S11c",
            "s": "S14a",
            "t": "S1e8",
            "u": "S115",
            "v": "S10e",
            "w": "S186",
            "x": "S10a",
            "y": "S19a",
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/file/dataset5/A',
                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/file/dataset5/B',
                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/file/dataset5/C',
                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/file/dataset5/D',
                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/file/dataset5/E',
                                                target_dir=f'{self.get_base_path()}/{self.name}/images')