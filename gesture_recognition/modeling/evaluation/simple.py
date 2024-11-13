from base.base.evaluation import tensorflow_evaluation, sklearn_evaluation, classification_report
from tensorflow.keras.models import load_model
from data import load
import joblib
import json
import os


def simple_evaluate(experiment_path, training_cfg):
    model_path = f"""{experiment_path}/model/{training_cfg.get("model")}/best_model"""
    if os.path.isdir(model_path):
        output_result_path = f"""{experiment_path}/evaluate/{training_cfg.get("model")}/simple"""
        os.makedirs(output_result_path, exist_ok=True)

        with open(f"{experiment_path}/dataset/test/info.json", 'r') as json_file:
            test_info = json.load(json_file)

        label_names = test_info["classes"].keys()

        dataset_test = load.load_ready_dataset(
            f"{experiment_path}/dataset/test/test.tfrecord")

        if training_cfg.get("framework") == "tensorflow":
            model = load_model(model_path)
            model_result = tensorflow_evaluation.evaluate_test_data(
                model, dataset_test)
        elif training_cfg.get("framework") == "sklearn":
            model = joblib.load(
                f"{model_path}/{training_cfg.get('model')}.joblib")
            model_result = sklearn_evaluation.evaluate_test_data(
                model, dataset_test)

        classification_report.report(output_result_path,
                                     model_result["y_true"],
                                     model_result["y_pred"],
                                     model_result["average_inference_time"],
                                     model_result["throughput"],
                                     save_results=True)

        classification_report.confusion_matrix_report(results_path=output_result_path,
                                                      y_true=model_result["y_true"],
                                                      y_pred=model_result["y_pred"])

        classification_report.normalized_confusion_matrix(results_path=output_result_path,
                                                          y_true=model_result["y_true"],
                                                          y_pred=model_result["y_pred"],
                                                          class_names=label_names)
