from base.base.training import callbacks, tensorflow_train, tensorflow_tuner
from base.hand_gesture import classification_models
from base.base.data import tensorflow_read
from base.base.utilities import config
from training import utils
from data import load
import json
import os


def tuner_model(experiment_path, training_cfg: dict, max_trials=30):

    utils.set_seed(42)

    model_path = f"""{experiment_path}/model/{training_cfg.get("model")}"""

    dataset_train = load.load_ready_dataset(
        f"{experiment_path}/dataset/train/train.tfrecord")
    dataset_val = load.load_ready_dataset(
        f"{experiment_path}/dataset/val/val.tfrecord")
    len_unique_classes, _ = tensorflow_read.count_unique_classes(dataset_val)

    dataset_train = dataset_train.batch(training_cfg.get("batch_size", 32))
    dataset_val = dataset_val.batch(training_cfg.get("batch_size", 32))
    callbacks_methods = [callbacks.scheduler_callback(training_cfg.get("learning_rate"),
                                                      training_cfg.get("lr_decay")),
                         callbacks.early_stop_callback(patience=20)]

    if training_cfg.get("model") == "fully_connected_embedder":
        best_hps = tensorflow_tuner.run_tuner(max_trials=max_trials,
                                              project_name="tuning",
                                              dataset_train=dataset_train,
                                              dataset_val=dataset_val,
                                              build_model_fn=classification_models.build_embedder_fc,
                                              num_classes=len_unique_classes, callbacks=callbacks_methods,
                                              save_dir=f"{model_path}/tuner/",
                                              epochs_tuner=100,
                                              args_epochs=100)

    elif training_cfg.get("model") == "fully_connected":
        best_hps = tensorflow_tuner.run_tuner(max_trials=max_trials,
                                              project_name="tuning",
                                              dataset_train=dataset_train,
                                              dataset_val=dataset_val,
                                              build_model_fn=classification_models.build_fc,
                                              num_classes=len_unique_classes, callbacks=callbacks_methods,
                                              save_dir=f"{model_path}/tuner/",
                                              epochs_tuner=30,
                                              args_epochs=100)

    print(best_hps.values)
    with open(f"{model_path}/params.json", "w") as params_file:
        json.dump(best_hps.values, params_file, indent=4)
