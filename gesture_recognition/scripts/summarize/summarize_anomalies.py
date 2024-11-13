import pandas as pd
import json
import yaml
import csv
import os

def extract_pipeline_info(pipeline_path):
    with open(pipeline_path, 'r') as file:
        pipeline_data = yaml.safe_load(file)
    
    # Extract required fields
    train_datasets = [entry['dataset'] for entry in pipeline_data.get('package', []) if entry.get('use') == 'train']
    test_datasets = [entry['dataset'] for entry in pipeline_data.get('package', []) if entry.get('use') == 'test']
    normalization = any("normalization" in entry.get("preprocess", {}).get("landmarks", []) for entry in pipeline_data.get('package', []))
    augmentation_methods = []
    augmentation_factor = 0
    augmentation = False
    sample = None
    
    for entry in pipeline_data.get('package', []):
        aug_methods = entry.get('augmentation', {}).get('methods', [])
        if "sample" in entry:
            sample = entry.get('sample', None)
            
        if aug_methods:
            augmentation = True
            augmentation_methods = aug_methods
            augmentation_factor = entry.get('augmentation', {}).get('factor', 0)
    
    return {
        "train_datasets": train_datasets,
        "test_datasets": test_datasets,
        "normalization": normalization,
        "augmentation": augmentation,
        "augmentation_factor": augmentation_factor,
        "augmentation_methods": augmentation_methods,
        "sample": sample
    }
    

def extract_anomaly_data(anomaly_report_path):
    with open(anomaly_report_path, 'r') as file:
        report_data = json.load(file)
    
    total_anomalies = sum([entry["anomalies"] for entry in report_data.values()])
    total_errors = sum([entry["error_model"] for entry in report_data.values()])
    error_anomalies = sum([entry["error_anomalies"] for entry in report_data.values()])
    anomalies_percent = sum([entry["anomaly_percentage"] for entry in report_data.values()]) / len(report_data)
    
    return {
        "anomalies": total_anomalies,
        "anomalies_percent": round(anomalies_percent, 2),
        "error_model": total_errors,
        "error_anomalies": error_anomalies
    }

def extract_anomaly_data(anomaly_report_path):
    with open(anomaly_report_path, 'r') as file:
        report_data = json.load(file)
    
    total_anomalies = sum([entry["anomalies"] for entry in report_data.values()])
    total_errors = sum([entry["error_model"] for entry in report_data.values()])
    error_anomalies = sum([entry["error_anomalies"] for entry in report_data.values()])
    anomalies_percent = sum([entry["anomaly_percentage"] for entry in report_data.values()]) / len(report_data)
    
    # Calculate the percentage of model errors that are also anomalies
    error_anomalies_percent = (error_anomalies / total_errors * 100) if total_errors > 0 else 0
    
    return {
        "anomalies": total_anomalies,
        "anomalies_percent": round(anomalies_percent, 2),
        "error_model": total_errors,
        "error_anomalies_percent": round(error_anomalies_percent, 2)
    }

def create_anomaly_summary_csv(experiment_folder, output_csv):
    anomaly_data = []

    for experiment_name in os.listdir(experiment_folder):
        experiment_path = os.path.join(experiment_folder, experiment_name)
        pipeline_path = os.path.join(experiment_path, 'pipeline.yml')
        
        if os.path.exists(pipeline_path):
            # Extract pipeline information
            pipeline_info = extract_pipeline_info(pipeline_path)
            
            for model in os.listdir(os.path.join(experiment_path, 'evaluate')):
                anomaly_path = os.path.join(experiment_path, 'evaluate', model, 'anomaly')
                if os.path.exists(anomaly_path):
                    for method in ["euclidean", "pca"]:
                        report_file = os.path.join(anomaly_path, f"{method}_anomaly_classification_report.json")
                        if os.path.exists(report_file):
                            # Extract anomaly data
                            anomaly_info = extract_anomaly_data(report_file)
                            
                            # Create a row for this anomaly method
                            anomaly_row = {
                                "experiment_name": experiment_name,
                                "train_datasets": ', '.join(pipeline_info["train_datasets"]),
                                "test_datasets": ', '.join(pipeline_info["test_datasets"]),
                                "normalization": pipeline_info["normalization"],
                                "augmentation": pipeline_info["augmentation"],
                                "augmentation_factor": pipeline_info["augmentation_factor"] if pipeline_info["augmentation"] else None,
                                "sample": pipeline_info["sample"],
                                "augmentation_methods": ', '.join(pipeline_info["augmentation_methods"]) if pipeline_info["augmentation"] else None,
                                "model": model,
                                "method": method,
                                "anomalies": anomaly_info["anomalies"],
                                "anomalies_percent": anomaly_info["anomalies_percent"],
                                "error_model": anomaly_info["error_model"],
                                "error_anomalies_percent": anomaly_info["error_anomalies_percent"]
                            }
                            anomaly_data.append(anomaly_row)

    # Convert all data to a DataFrame and save it to a CSV
    anomaly_summary_df = pd.DataFrame(anomaly_data)
    anomaly_summary_df.to_csv(output_csv, index=False)
    print(f"Anomaly summary CSV saved at: {output_csv}")
    
# Usage
experiment_folder = "assets/experiment_data/"
anomaly_output_csv = "global_anomaly_summary.csv"
create_anomaly_summary_csv(experiment_folder, anomaly_output_csv)