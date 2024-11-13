from dataclasses import dataclass
from dataset import Dataset

@dataclass
class PSL(Dataset):
    
    name: str = "PSL"
    
    def download(self):
        #https://www.sciencedirect.com/science/article/pii/S235234092100305X
        self.download_from_url("https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/y9svrbh27n-1.zip")
        return self
    
    def get_mapper(self):
        return {
            "Ain ع": "S1ec",
           "Aliph ا":"S1f8",
           "aRay ڑ":"S10e",
           "Bari yeh ے":"S192",
           "Bay ب":"S147",
           "Chay چ":"S18c",
           "Chhoti yeh ی":"S14a",
           "Daal  د":"S10a",
           "Daal ڈ":"S101",
           "Dhaal ذ":"S118",
           "Dhuaad ض":"S1a0",
           "Djay ژ":"S12e",
           "Fay ف":"S1d8",
           "Gaaf گ":"S100",
           "Ghain غ":"S135",
           "Hamza ‍‌ء":"S1f2",
           "Hay ہ":"S119",
           "hey ح":"S115",
           "Jeem ج":"S192",
           "Kaaf ک":"S10e",
           "Khay خ":"S11e",
           "Laam ل":"S1dc",
           "Meem م":"S18d",
           "Noon ن":"S13f",
           "Pay پ":"S10e",
           "Quaaf ق":"S12b",
           "Ray ر":"S11a",
           "Seen س":"S203",
           "Sheen ‎‎ش":"S13f",
           "Suaad ص":"S1f5",
           "Tay ت":"S119",
           "Tey ٹ":"S1fb",
           "Thay ث":"S16d",
           "Toay'n ط":"S103",
           "Vao و":"S1f9",
           "Zay ز":"S100",
           "Zoay'n ظ":"S1a3"
        }
    
    def map_classes(self):
        self.map_classes_to_sign_writing_format(source_dir=f'{self.get_base_path()}/{self.name}/original/PSL',
                                                                target_dir=f'{self.get_base_path()}/{self.name}/images')
        