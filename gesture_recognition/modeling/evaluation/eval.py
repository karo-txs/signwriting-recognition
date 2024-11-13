from base.base.evaluation import tensorflow_evaluation, sklearn_evaluation, classification_report
from tensorflow.keras.models import load_model
from evaluation import bootstrap, anomaly, error, simple
from data import load
import joblib
import json
import os


def evaluate_model(experiment_path, training_cfg, eval_method, base_data_path, package_cfg):
    if eval_method == "simple":
        simple.simple_evaluate(experiment_path, training_cfg)
    elif eval_method == "bootstrap":
        bootstrap.bootstrap_evaluate(experiment_path, training_cfg)
    elif eval_method == "anomaly":
        anomaly.landmarks_anomaly(experiment_path, training_cfg)
    elif eval_method == "error":
        error.error_analysis(base_data_path, package_cfg,
                             experiment_path, training_cfg)
