from dataclasses import dataclass
import zipfile
from dataset import Dataset

@dataclass
class ArSL21L(Dataset):
    
    name: str = "ArSL21L"
    
    def download(self):
        self.download_from_kaggle("ammarsayedtaha/arabic-sign-language-dataset-2022")
        return self
    
    def get_mapper(self):
        prefixes = [
        "3_55_F_", "11_16_M_", "14_19_F_", "23_19_M_", 
        "831_23_M_", "1003_19_F_", "1005_16_F_", "1011_37_F_", 
        "1017_35_M_", "1031_44_F_", "1034_19_M_", "1035_18_F_", 
        "1051_25_M_", "1055_68_F_", "1058_69_M_"
    ]
        
        mappings = {
            "ain_": "S115",
            "al_": "S11e",
            "aleff_": "S1f5",
            "bb_": "S100",
            "dal_": "S1ed",
            "dha_": "S140",
            "dhad_": "S1f5",
            "fa_": "S1eb",
            "gaaf_": "S12a",
            "ghain_": "S12d",
            "ha_": "S177",
            "haa_": "S185",
            "jeem_": "S16d",
            "kaaf_": "S147",
            "khaa_": "S182",
            "la_": "S1a0",
            "laam_": "S1dc",
            "meem_": "S192",
            "nun_": "S1ed",
            "ra_": "S100",
            "saad_": "S1f8",
            "seen_": "S15a",
            "sheen_": "S144",
            "ta_": "S115",
            "taa_": "S142",
            "thaa_": "S18c",
            "toot_": "S110",
            "waw_": "S1fa", 
            "ya_": "S1dc",
            "yaa_": "S19a",
        }
        result = {}
        
        for key, value in mappings.items():
            for prefix in prefixes:
                result[f"{prefix}{key}"] = value
                
        return result
    
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(source_dir=f'{self.get_base_path()}/{self.name}/original/datasets/valid/images',
                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        