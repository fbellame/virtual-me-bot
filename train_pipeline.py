import os
import json
import zipfile
import subprocess
from pathlib import Path

# Paths
QUEUE_FOLDER = "queue"
DATASET_DIR = "/media/farid/data1/projects/ai-toolkit/dataset"
SCRIPT_PATH = "/path/to/start_aitoolkit.sh"  # Update with the actual path to the script

def process_file(file_path):
    try:
        # Read the JSON file
        with open(file_path, 'r') as f:
            data = json.load(f)

        # Extract information
        dataset_zip = data.get("dataset_zip")
        yaml_path = data.get("yaml_path")
        status = data.get("status")

        # Validate the status
        if status != "queued":
            print(f"Skipping file {file_path} with status {status}")
            return

        # Unzip the dataset
        zip_path = Path(dataset_zip)
        if zip_path.is_file():
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DATASET_DIR)
            print(f"Extracted {dataset_zip} to {DATASET_DIR}")
        else:
            print(f"Dataset zip file {dataset_zip} not found.")
            return

        # Run the script with the YAML file as the parameter
        subprocess.run(["bash", SCRIPT_PATH, yaml_path], check=True)
        print(f"Executed {SCRIPT_PATH} with parameter {yaml_path}")

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

if __name__ == "__main__":
    # Process all JSON files in the queue folder
    for json_file in Path(QUEUE_FOLDER).glob("*.json"):
        print(f"Processing {json_file}...")
        process_file(json_file)
