from core.train.steps.train_model.frameworks.tensorflow_train import TensorflowTrain
from core.train.intefaces.tf_model import TFModel
from core.train.intefaces.model import Model


def get_trainer_model(model: Model):
    trainer = None

    if isinstance(model, TFModel):
        trainer = TensorflowTrain(model)

    return trainer
