from core.train.intefaces.model import Model
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Trainer(ABC):
    model: Model

    @abstractmethod
    def train(self, dataset_train, dataset_val):
        pass
