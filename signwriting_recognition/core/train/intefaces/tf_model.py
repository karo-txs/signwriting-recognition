from core.train.intefaces.model import Model
from dataclasses import dataclass, field
import logging


@dataclass
class TFModel(Model):
    learning_rate: float = field(default=None)
    lr_decay: float = field(default=None)
    epochs: int = field(default=None)

    def summary(self):
        logging.info(self.builded_model.summary())
