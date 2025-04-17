import tensorflow as tf
import os


def scheduler_callback(learning_rate: float, lr_decay: float):
    def scheduler(epoch): return learning_rate * (lr_decay**epoch)
    return tf.keras.callbacks.LearningRateScheduler(scheduler)


def checkpoint_callback(save_model_dir: str):
    checkpoint_path = os.path.join(
        f"{save_model_dir}/checkpoints", 'model-{epoch:04d}')
    return tf.keras.callbacks.ModelCheckpoint(
        checkpoint_path,
        save_weights_only=False,
    )


def best_checkpoint_callback(save_model_dir: str):
    best_model_path = os.path.join(save_model_dir, 'best_model')
    return tf.keras.callbacks.ModelCheckpoint(
        best_model_path,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True,
        save_weights_only=False,
    )


def tensorboard_callback(save_model_dir: str):
    return tf.keras.callbacks.TensorBoard(
        log_dir=os.path.join(save_model_dir, 'tensorboard'))


def early_stop_callback(patience: int = 30):
    return tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=patience)
