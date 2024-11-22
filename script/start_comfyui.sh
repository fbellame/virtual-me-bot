#!/bin/bash

# Activate the Conda environment
source "$(conda info --base)/etc/profile.d/conda.sh" || { echo "Failed to source Conda!"; exit 1; }
conda activate comfyui || { echo "Failed to activate Conda environment 'comfyui'!"; exit 1; }

# Navigate to the ComfyUI installation directory
cd /media/farid/data1/projects/ComfyUI || { echo "ComfyUI directory not found!"; exit 1; }

# Start ComfyUI in the background
nohup python main.py > comfyui.log 2>&1 &

# Notify the user
echo "ComfyUI has been started. Logs are being written to comfyui.log."
