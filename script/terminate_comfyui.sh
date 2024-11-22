#!/bin/bash

# Define the script name to search for
SCRIPT_NAME="python main.py"

# Find the process ID of the target script
PID=$(ps -ef | grep "$SCRIPT_NAME" | grep -v "grep" | awk '{print $2}')

# Check if the process ID is found
if [ -z "$PID" ]; then
    echo "Process '$SCRIPT_NAME' not found."
    exit 1
fi

# Display the process ID
echo "Found process '$SCRIPT_NAME' with PID: $PID"


# Terminate the process
kill "$PID"
if [ $? -eq 0 ]; then
    echo "Process '$SCRIPT_NAME' with PID $PID terminated successfully."
else
    echo "Failed to terminate the process '$SCRIPT_NAME' with PID $PID."
fi
