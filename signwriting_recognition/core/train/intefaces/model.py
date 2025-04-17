from dataclasses import dataclass, field
from abc import ABC, abstractmethod


@dataclass
class Model(ABC):
    models_path: str
    model_name: str = field(default=None)
    builded_model: any = field(default=None)

    @abstractmethod
    def build_from_dict(self, params: dict, len_unique_classes: int):
        pass
