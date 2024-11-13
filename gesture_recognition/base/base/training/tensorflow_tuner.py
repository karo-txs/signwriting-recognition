import keras_tuner as kt
import tensorflow as tf


def model_builder(hp, build_model_fn, num_classes):
    args_dropout = hp.Float('dropout', min_value=0.01,
                            max_value=0.5, step=0.01)
    args_learning_rate = hp.Choice('learning_rate', values=[
                                   1e-1, 1e-2, 1e-3, 1e-4])

    args_units = hp.Int('fc_units', min_value=12, max_value=1024, step=12)
    args_layers = hp.Int('fc_layers', min_value=1, max_value=12, step=1)

    return build_model_fn(fc_layers=args_layers, fc_units=args_units,
                          dropout=args_dropout, num_classes=num_classes,
                          learning_rate=args_learning_rate)


def run_tuner(max_trials, project_name, dataset_train,
              dataset_val, build_model_fn, num_classes, callbacks,
              save_dir,
              epochs_tuner=20, args_epochs=100):
    tuner = kt.BayesianOptimization(
        lambda hp: model_builder(hp, build_model_fn, num_classes),
        objective='val_accuracy',
        max_trials=max_trials,
        seed=42,
        directory=save_dir,
        project_name=project_name,
        num_initial_points=2
    )

    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)

    tuner.search(x=dataset_train,
                 epochs=epochs_tuner,
                 verbose=1,
                 validation_data=dataset_val,
                 callbacks=[stop_early])

    # Get the optimal hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps
