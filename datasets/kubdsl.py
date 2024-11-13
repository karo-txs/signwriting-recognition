from dataclasses import dataclass
from dataset import Dataset

@dataclass
class KUBsSL(Dataset):
    
    name: str = "KUBsSL"
    
    def download(self):
        self.download_from_url("https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/scpvm2nbkm-4.zip")
        return self
    
    def get_mapper(self):
        return {
            "2433": "S1dc",
            "2434": "S1f4",
            "2435": "S176",
            "2453": "S16d",
            "2454": "S173",
            "2455": "S185",
            "2456": "S1eb",
            "2457": "S1ce",
            "2458": "S1ce",
            "2459": "S1a1",
            "2460-2479": "S150",
            "2461": "S12d",
            "2462": "S1d5",
            "2463": "S1fb",
            "2464": "S1f4",
            "2465": "S124",
            "2466": "S105",
            "2467-2472": "S1b1",
            "2468-2510": "S201",
            "2469": "S1dc",
            "2470": "S1f5",
            "2471": "S168",
            "2474": "S10e",
            "2475": "S1ce",
            "2476-2477": "S147",
            "2478": "S18c",
            "2480-2524-2525": "S11a",
            "2482": "S1dc",
            "2486-2488-2487": "S1ff",
            "2489": "S115"
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/KU-BdSL Khulna University Bengali Sign Language dataset/KU-BdSL/MSLD',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        