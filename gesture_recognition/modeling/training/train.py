from base.base.training import callbacks, tensorflow_train, sklearn_train
from base.hand_gesture import classification_models
from base.base.data import tensorflow_read
from training import utils
from data import load
import json
import os


def load_params_from_json(model_path):
    params_path = os.path.join(model_path, "params.json")
    if os.path.exists(params_path):
        with open(params_path, "r") as file:
            params = json.load(file)
        print(f"Carregando parâmetros do JSON: {params}")
        return params
    return {}


def train_model(experiment_path, training_cfg: dict):
    model_path = f"""{experiment_path}/model/{training_cfg.get("model")}"""

    utils.set_seed(42)

    if training_cfg.get("model") in ["fully_connected", "fully_connected_embedder"]:
        params = load_params_from_json(model_path)

        if params:
            training_cfg["dropout"] = params["dropout"]
            training_cfg["learning_rate"] = params["learning_rate"]
            training_cfg["fc_units"] = params["fc_units"]
            training_cfg["fc_layers"] = params["fc_layers"]

    dataset_train = load.load_ready_dataset(
        f"{experiment_path}/dataset/train/train.tfrecord")
    dataset_val = load.load_ready_dataset(
        f"{experiment_path}/dataset/val/val.tfrecord")
    len_unique_classes, _ = tensorflow_read.count_unique_classes(dataset_val)

    if training_cfg.get("model") == "fully_connected_embedder":
        model = classification_models.build_embedder_fc(unfrozen_layers=training_cfg.get("unfrozen_layers"),
                                                        fc_layers=training_cfg.get(
                                                            "fc_layers"),
                                                        fc_units=training_cfg.get(
                                                            "fc_units"),
                                                        dropout=training_cfg.get(
                                                            "dropout"),
                                                        num_classes=len_unique_classes,
                                                        learning_rate=training_cfg.get("learning_rate"))

    elif training_cfg.get("model") == "fully_connected":
        model = classification_models.build_fc(fc_layers=training_cfg.get("fc_layers"),
                                               fc_units=training_cfg.get(
                                                   "fc_units"),
                                               dropout=training_cfg.get(
                                                   "dropout"),
                                               num_classes=len_unique_classes,
                                               learning_rate=training_cfg.get("learning_rate"))

    elif training_cfg.get("model") == "fully_connected_2":
        model = classification_models.build_fc_2(fc_layers=training_cfg.get("fc_layers"),
                                                 fc_units=training_cfg.get(
                                                     "fc_units"),
                                                 dropout=training_cfg.get(
                                                     "dropout"),
                                                 num_classes=len_unique_classes,
                                                 learning_rate=training_cfg.get("learning_rate"))

    elif training_cfg.get("model") == "conv_fully_connected":
        model = classification_models.build_conv_fc(fc_layers=training_cfg.get("fc_layers"),
                                                    fc_units=training_cfg.get(
                                                        "fc_units"),
                                                    dropout=training_cfg.get(
                                                        "dropout"),
                                                    num_classes=len_unique_classes,
                                                    learning_rate=training_cfg.get("learning_rate"))

    elif training_cfg.get("model") == "conv1d":
        model = classification_models.build_conv(fc_layers=training_cfg.get("fc_layers"),
                                                 fc_units=training_cfg.get(
                                                     "fc_units"),
                                                 dropout=training_cfg.get(
                                                     "dropout"),
                                                 conv_layers=4,
                                                 conv_filters=64,
                                                 num_classes=len_unique_classes,
                                                 learning_rate=training_cfg.get("learning_rate"))

    elif training_cfg.get("model") == "gnn":
        model = classification_models.build_gnn(num_classes=len_unique_classes,
                                                learning_rate=training_cfg.get(
                                                    "learning_rate"),
                                                gnn_layers=2,
                                                gnn_units=64,
                                                fc_layers=training_cfg.get(
                                                    "fc_layers"),
                                                fc_units=training_cfg.get(
                                                    "fc_units"),
                                                dropout=training_cfg.get("dropout"))

    elif training_cfg.get("model") == "lstm":
        model = classification_models.build_lstm(lstm_units=64,
                                                 lstm_layers=2,
                                                 fc_layers=training_cfg.get(
                                                     "fc_layers"),
                                                 fc_units=training_cfg.get(
                                                     "fc_units"),
                                                 dropout=training_cfg.get(
                                                     "dropout"),
                                                 num_classes=len_unique_classes,
                                                 learning_rate=training_cfg.get("learning_rate"))
    elif training_cfg.get("model") == "random_forest":
        model = classification_models.build_random_forest()

    elif training_cfg.get("model") == "svm":
        model = classification_models.build_svm()

    elif training_cfg.get("model") == "gbc":
        model = classification_models.build_gbc()

    elif training_cfg.get("model") == "adaboost":
        model = classification_models.build_adaboost()

    elif training_cfg.get("model") == "knn":
        model = classification_models.build_knn()

    elif training_cfg.get("model") == "lr":
        model = classification_models.build_lr()

    if training_cfg.get("framework") == "tensorflow":
        dataset_train = dataset_train.batch(training_cfg.get("batch_size", 32))
        dataset_val = dataset_val.batch(training_cfg.get("batch_size", 32))

        print(model.summary())
        callbacks_methods = [callbacks.best_checkpoint_callback(model_path),
                             callbacks.scheduler_callback(training_cfg.get("learning_rate"),
                                                          training_cfg.get("lr_decay")),
                             callbacks.early_stop_callback(patience=20),
                             callbacks.tensorboard_callback(model_path)]

        history, model = tensorflow_train.train_model(model, dataset_train, dataset_val,
                                                      callbacks_methods, epochs=training_cfg.get("epochs", 100))

        print(history)
        print(model.evaluate(dataset_val, verbose=0))
    elif training_cfg.get("framework") == "sklearn":
        model_path = f"{model_path}/best_model/"
        os.makedirs(model_path, exist_ok=True)
        model = sklearn_train.train_model(
            model, dataset_train, dataset_val, f"{model_path}/{training_cfg.get('model')}.joblib")
