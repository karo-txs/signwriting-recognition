import numpy as np
import json
import csv
import os


def summarize(experiment_path, training_cfg, eval_method):
    if eval_method == "bootstrap":
        summarize_bootstrap(experiment_path, training_cfg)
    elif eval_method == "simple":
        summarize_simple(experiment_path, training_cfg)
    
def summarize_bootstrap(experiment_path, training_cfg):
    summary_file = os.path.join(experiment_path, "evaluate", "bootstrap_summary.csv")
    header = [
        "model", "average_inference_time", "min_inference_time", "max_inference_time",
        "average_throughput", "min_throughput", "max_throughput",
        "average_accuracy", "min_accuracy", "max_accuracy", "ci_accuracy_low", "ci_accuracy_high", "margin_error_accuracy",
        "average_precision", "min_precision", "max_precision", "ci_precision_low", "ci_precision_high", "margin_error_precision",
        "average_recall", "min_recall", "max_recall", "ci_recall_low", "ci_recall_high", "margin_error_recall",
        "average_f1_score", "min_f1_score", "max_f1_score", "ci_f1_score_low", "ci_f1_score_high", "margin_error_f1_score"
    ]

    with open(summary_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()

        for model in training_cfg:
            result_path = os.path.join(experiment_path, "evaluate", model.get("model"), "bootstrap", "bootstrap_aggregated_results.json")
            iteration_files_path = os.path.join(experiment_path, "evaluate", model.get("model"), "bootstrap")
            
            # Load aggregated results
            if os.path.exists(result_path):
                with open(result_path, 'r') as json_file:
                    aggregated_results = json.load(json_file)
            else:
                print(f"Warning: Aggregated result file not found for model {model.get('model')} at {result_path}")
                continue
            
            # Load each bootstrap iteration
            metrics = ["accuracy", "precision", "recall", "f1_score"]
            data = {metric: [] for metric in metrics}

            for iteration_file in os.listdir(iteration_files_path):
                if iteration_file.startswith("bootstrap_iteration_") and iteration_file.endswith(".json"):
                    with open(os.path.join(iteration_files_path, iteration_file), 'r') as iter_file:
                        iteration_data = json.load(iter_file)
                        for metric in metrics:
                            data[metric].append(iteration_data[metric])

            # Calculate confidence intervals and errors for each metric
            row = {"model": model.get("model")}
            row.update({key: round(value, 4) for key, value in aggregated_results.items()})

            for metric in metrics:
                ci_low = np.percentile(data[metric], 2.5)
                ci_high = np.percentile(data[metric], 97.5)
                margin_of_error = (ci_high - ci_low) / 2

                row[f"ci_{metric}_low"] = round(ci_low, 4)
                row[f"ci_{metric}_high"] = round(ci_high, 4)
                row[f"margin_error_{metric}"] = round(margin_of_error, 4)

            writer.writerow(row)

    print(f"Summary CSV with confidence intervals and margin of error saved at: {summary_file}")
    

def summarize_simple(experiment_path, training_cfg):
    summary_file = os.path.join(experiment_path, "evaluate", "simple_summary.csv")
    header = [
        "model", "accuracy", "recall_macro_avg", "precision_macro_avg", "f1_macro_avg",
        "recall_weighted", "precision_weighted", "f1_weighted",
        "average_inference_time", "throughput"
    ]

    with open(summary_file, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()

        for model in training_cfg:
            report_path = os.path.join(experiment_path, "evaluate", model.get("model"), "simple", "report.csv")
            
            if os.path.exists(report_path):
                with open(report_path, 'r') as report_file:
                    reader = csv.DictReader(report_file)
                    for row in reader:
                        row = {key: (round(float(value), 4) if key != "model" else value) for key, value in row.items()}
                        row["model"] = model.get("model")
                        writer.writerow(row)
            else:
                print(f"Warning: Report file not found for model {model.get('model')} at {report_path}")

    print(f"Summary CSV saved at: {summary_file}")
