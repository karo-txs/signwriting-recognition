from core.train.steps.train_model.frameworks import tensorflow_callbacks
from core.train.intefaces.trainer import Trainer
import tensorflow as tf
import logging


class TensorflowTrain(Trainer):

    def _split_validation(self, ds: tf.data.Dataset, val_fraction: float = 0.10):
        """
        Cria `ds_train` e `ds_val` (~val_fraction) usando enumeração,
        sem depender da cardinalidade.
        """
        if not 0.0 < val_fraction < 1.0:
            raise ValueError("val_fraction deve estar em (0, 1).")

        # Ex.: 0.10 → cada 10‑ª amostra vai para validação
        mod = int(round(1 / val_fraction))

        # enumerate → (idx, sample).  idx % mod == 0 → validação
        ds_en = ds.enumerate()

        ds_val = (
            ds_en.filter(lambda i, _: tf.equal(i % mod, 0))
            .map(lambda _, x: x)  # remove índice
            .batch(self.model.batch_size)
        )
        ds_train = (
            ds_en.filter(lambda i, _: tf.not_equal(i % mod, 0))
            .map(lambda _, x: x)
            .batch(self.model.batch_size)
        )

        logging.info(
            f"Split automático de validação (~{val_fraction:.0%}) "
            f"usando i % {mod} == 0."
        )
        return ds_train, ds_val

    def train(
        self, dataset_train: tf.data.Dataset, dataset_val: tf.data.Dataset | None = None
    ):
        if dataset_val is None:
            dataset_train, dataset_val = self._split_validation(dataset_train, 0.10)
        else:
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

        val_results = self.model.builded_model.evaluate(dataset_val, verbose=0)
        logging.info(f"TrainPipeline: Evaluate Results - {val_results}")

        return history, self.model
