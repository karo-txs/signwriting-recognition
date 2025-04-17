from core.train.steps.train_model.frameworks import tensorflow_callbacks
from core.train.intefaces.trainer import Trainer
import logging


class TensorflowTrain(Trainer):

    def train(self, dataset_train, dataset_val):
        dataset_train = dataset_train.batch(self.model.batch_size)
        dataset_val = dataset_val.batch(self.model.batch_size)
        
        callbacks_methods = [
            tensorflow_callbacks.best_checkpoint_callback(self.model.models_path),
            tensorflow_callbacks.scheduler_callback(
                self.model.learning_rate, self.model.lr_decay
            ),
            tensorflow_callbacks.early_stop_callback(patience=20),
            tensorflow_callbacks.tensorboard_callback(self.model.models_path),
        ]

        history = self.model.builded_model.fit(
            x=dataset_train,
            epochs=self.model.epochs,
            validation_data=dataset_val,
            callbacks=callbacks_methods,
        )
        
        logging.info(f"TrainPipeline: Evaluate Results - {self.model.builded_model.evaluate(dataset_val, verbose=0)}")

        return history, self.model
