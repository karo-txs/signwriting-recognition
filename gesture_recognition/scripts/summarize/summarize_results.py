import os
import yaml
import csv
import pandas as pd

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

def extract_bootstrap_summary(bootstrap_summary_path):
    return pd.read_csv(bootstrap_summary_path)

def create_experiment_summary_csv(experiment_folder, output_csv):
    experiment_data = []

    for experiment_name in os.listdir(experiment_folder):
        experiment_path = os.path.join(experiment_folder, experiment_name)
        pipeline_path = os.path.join(experiment_path, 'pipeline.yml')
        bootstrap_summary_path = os.path.join(experiment_path, 'evaluate', 'bootstrap_summary.csv')
        
        if os.path.exists(pipeline_path) and os.path.exists(bootstrap_summary_path):
            # Extract information from pipeline.yml
            pipeline_info = extract_pipeline_info(pipeline_path)
            
            # Extract information from bootstrap_summary.csv
            bootstrap_summary = extract_bootstrap_summary(bootstrap_summary_path)
            
            # Add experiment name and pipeline info to each row of bootstrap summary
            for _, row in bootstrap_summary.iterrows():
                experiment_row = {
                    "experiment_name": experiment_name,
                    "train_datasets": ', '.join(pipeline_info["train_datasets"]),
                    "test_datasets": ', '.join(pipeline_info["test_datasets"]),
                    "normalization": pipeline_info["normalization"],
                    "augmentation": pipeline_info["augmentation"],
                    "augmentation_factor": pipeline_info["augmentation_factor"] if pipeline_info["augmentation"] else None,
                    "sample": pipeline_info["sample"],
                    "augmentation_methods": ', '.join(pipeline_info["augmentation_methods"]) if pipeline_info["augmentation"] else None,
                }
                experiment_row.update(row.to_dict())
                experiment_data.append(experiment_row)

    # Convert all data to a DataFrame and save it to a CSV
    experiment_summary_df = pd.DataFrame(experiment_data)
    experiment_summary_df.to_csv(output_csv, index=False)
    print(f"Global experiment summary CSV saved at: {output_csv}")

# Usage
experiment_folder = "assets/experiment_data/"
output_csv = "global_experiment_summary.csv"
create_experiment_summary_csv(experiment_folder, output_csv)
