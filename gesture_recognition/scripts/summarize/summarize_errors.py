import pandas as pd
import json
import yaml
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

def extract_error_summary(error_summary_path):
    with open(error_summary_path, 'r') as file:
        error_data = json.load(file)
    
    return {
        "model_error": error_data["model_error"],
        "mediapipe_error": error_data["mediapipe_error"],
        "mediapipe_not_found_error": error_data["mediapipe_not_found_error"],
        "total_error": error_data["total_error"],
        "total_error_without_not_found": error_data["total_error_without_not_found"],
        "total_samples": error_data["total_samples"],
        "adjusted_accuracy": error_data["adjusted_accuracy"],
        "original_accuracy": error_data["original_accuracy"]
    }

def create_error_summary_csv(experiment_folder, output_csv):
    error_data = []

    for experiment_name in os.listdir(experiment_folder):
        experiment_path = os.path.join(experiment_folder, experiment_name)
        pipeline_path = os.path.join(experiment_path, 'pipeline.yml')
        
        if os.path.exists(pipeline_path):
            # Extract pipeline information
            pipeline_info = extract_pipeline_info(pipeline_path)
            
            for model in os.listdir(os.path.join(experiment_path, 'evaluate')):
                error_analysis_path = os.path.join(experiment_path, 'evaluate', model, 'error_analysis')
                error_summary_file = os.path.join(error_analysis_path, 'error_summary.json')
                
                if os.path.exists(error_summary_file):
                    # Extract error summary data
                    error_summary = extract_error_summary(error_summary_file)
                    
                    # Create a row for this error analysis
                    error_row = {
                        "experiment_name": experiment_name,
                        "train_datasets": ', '.join(pipeline_info["train_datasets"]),
                        "test_datasets": ', '.join(pipeline_info["test_datasets"]),
                        "normalization": pipeline_info["normalization"],
                        "augmentation": pipeline_info["augmentation"],
                        "augmentation_factor": pipeline_info["augmentation_factor"] if pipeline_info["augmentation"] else None,
                        "sample": pipeline_info["sample"],
                        "augmentation_methods": ', '.join(pipeline_info["augmentation_methods"]) if pipeline_info["augmentation"] else None,
                        "model": model,
                        "model_error": error_summary["model_error"],
                        "mediapipe_error": error_summary["mediapipe_error"],
                        "mediapipe_not_found_error": error_summary["mediapipe_not_found_error"],
                        "total_error": error_summary["total_error"],
                        "total_error_without_not_found": error_summary["total_error_without_not_found"],
                        "total_samples": error_summary["total_samples"],
                        "adjusted_accuracy": error_summary["adjusted_accuracy"],
                        "original_accuracy": error_summary["original_accuracy"]
                    }
                    error_data.append(error_row)

    # Convert all data to a DataFrame and save it to a CSV
    error_summary_df = pd.DataFrame(error_data)
    error_summary_df.to_csv(output_csv, index=False)
    print(f"Error summary CSV saved at: {output_csv}")
    
# Usage
experiment_folder = "assets/experiment_data/"
error_output_csv = "global_error_summary.csv"
create_error_summary_csv(experiment_folder, error_output_csv)
