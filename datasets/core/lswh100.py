from interface.dataset import Dataset
from dataclasses import dataclass


@dataclass
class LSWH100(Dataset):

    name: str = "LSWH100"

    def download(self):
        print(
            "Não é possível fazer o download do dataset hospedado no Zenodo. \n"
            + "Baixe o dataset a partir do link <https://zenodo.org/records/10635628> \n"
            + "e salve na pasta LSWH100/original já descompactado"
        )
        return self

    def get_mapper(self):
        return {
            "S15a": "S15a",
            "S203": "S203",
            "S100": "S100",
            "S14c": "S14c",
            "S15d": "S15d",
            "S1f7": "S1f7",
            "S1dc": "S1dc",
            "S176": "S176",
            "S16d": "S16d",
            "S10e": "S10e",
            "S115": "S115",
            "S192": "S192",
            "S11a": "S11a",
            "S140": "S140",
            "S1f5": "S1f5",
            "S19a": "S19a",
            "S101": "S101",
            "S147": "S147",
            "S153": "S153",
            "S14a": "S14a",
            "S1ed": "S1ed",
            "S1ce": "S1ce",
            "S150": "S150",
            "S1ea": "S1ea",
            "S119": "S119",
            "S157": "S157",
            "S177": "S177",
            "S144": "S144",
            "S185": "S185",
            "S1d2": "S1d2",
            "S110": "S110",
            "S10a": "S10a",
            "S18d": "S18d",
            "S1d3": "S1d3",
            "S11e": "S11e",
            "S180": "S180",
            "S186": "S186",
            "S10b": "S10b",
            "S1de": "S1de",
            "S1f4": "S1f4",
            "S118": "S118",
            "S18c": "S18c",
            "S106": "S106",
            "S17d": "S17d",
            "S1d4": "S1d4",
            "S1c5": "S1c5",
            "S16c": "S16c",
            "S1f8": "S1f8",
            "S182": "S182",
            "S12d": "S12d",
            "S154": "S154",
            "S1ec": "S1ec",
            "S181": "S181",
            "S1a0": "S1a0",
            "S1f0": "S1f0",
            "S1bb": "S1bb",
            "S142": "S142",
            "S175": "S175",
            "S1ee": "S1ee",
            "S1eb": "S1eb",
            "S16f": "S16f",
            "S187": "S187",
            "S10c": "S10c",
            "S1fa": "S1fa",
            "S1df": "S1df",
            "S1ef": "S1ef",
            "S1f1": "S1f1",
            "S14e": "S14e",
            "S17e": "S17e",
            "S19c": "S19c",
            "S1f2": "S1f2",
            "S152": "S152",
            "S133": "S133",
            "S171": "S171",
            "S1e4": "S1e4",
            "S1c3": "S1c3",
            "S128": "S128",
            "S13f": "S13f",
            "S17f": "S17f",
            "S160": "S160",
            "S13d": "S13d",
            "S173": "S173",
            "S1c1": "S1c1",
            "S17c": "S17c",
            "S1d0": "S1d0",
            "S1d1": "S1d1",
            "S18b": "S18b",
            "S1a8": "S1a8",
            "S127": "S127",
            "S1a7": "S1a7",
            "S1da": "S1da",
            "S151": "S151",
            "S1f9": "S1f9",
            "S194": "S194",
            "S155": "S155",
            "S174": "S174",
            "S1a3": "S1a3",
            "S198": "S198",
            "S1fb": "S1fb",
            "S1a5": "S1a5",
        }

    def map_classes(self):
        self.map_classes_to_sign_writing_format_file_name_based(
            source_dir=[
                f"{self.get_base_path()}/{self.name}/original/back/test",
                f"{self.get_base_path()}/{self.name}/original/front/test",
                f"{self.get_base_path()}/{self.name}/original/left/test",
                f"{self.get_base_path()}/{self.name}/original/right/test",
            ],
            target_dir=f"{self.get_base_path()}/{self.name}/sw-classified/test",
            not_ignore_exists=False,
        )
        self.map_classes_to_sign_writing_format_file_name_based(
            source_dir=[
                f"{self.get_base_path()}/{self.name}/original/back/train",
                f"{self.get_base_path()}/{self.name}/original/front/train",
                f"{self.get_base_path()}/{self.name}/original/left/train",
                f"{self.get_base_path()}/{self.name}/original/right/train",
            ],
            target_dir=f"{self.get_base_path()}/{self.name}/sw-classified/train",
            not_ignore_exists=False,
            recursive=True
        )
        self.map_classes_to_sign_writing_format_file_name_based(
            source_dir=[
                f"{self.get_base_path()}/{self.name}/original/back/val",
                f"{self.get_base_path()}/{self.name}/original/front/val",
                f"{self.get_base_path()}/{self.name}/original/left/val",
                f"{self.get_base_path()}/{self.name}/original/right/val",
            ],
            target_dir=f"{self.get_base_path()}/{self.name}/sw-classified/val",
            not_ignore_exists=False,
        )
