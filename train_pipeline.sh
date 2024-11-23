#!/bin/bash

# Configuration
QUEUE_FOLDER="queue"
DATASET_DIR="/media/farid/data1/projects/ai-toolkit/dataset"
SCRIPT_PATH="/path/to/start_aitoolkit.sh"  # Update with the actual path to the script

# Function to process a single JSON file
process_file() {
    local file_path="$1"

    echo "Processing $file_path..."

    # Read and parse JSON fields
    dataset_zip=$(jq -r '.dataset_zip' "$file_path")
    yaml_path=$(jq -r '.yaml_path' "$file_path")
    status=$(jq -r '.status' "$file_path")

    # Check if status is "queued"
    if [[ "$status" != "queued" ]]; then
        echo "Skipping $file_path with status $status"
        return
    fi

    # Check if dataset_zip exists
    if [[ -f "$dataset_zip" ]]; then
        echo "Extracting $dataset_zip to $DATASET_DIR..."
        unzip -o "$dataset_zip" -d "$DATASET_DIR"
    else
        echo "Dataset zip file $dataset_zip not found. Skipping."
        return
    fi

    # Run the script with yaml_path as the first parameter
    if [[ -f "$SCRIPT_PATH" ]]; then
        echo "Executing $SCRIPT_PATH with parameter $yaml_path..."
        bash "$SCRIPT_PATH" "$yaml_path"
    else
        echo "Script $SCRIPT_PATH not found. Skipping."
        return
    fi
}

# Main loop to process all JSON files in the queue folder
for json_file in "$QUEUE_FOLDER"/*.json; do
    if [[ -f "$json_file" ]]; then
        process_file "$json_file"
    else
        echo "No JSON files found in $QUEUE_FOLDER."
    fi
done
