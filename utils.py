
import requests
import os
import base64
import subprocess

DATASET_FOLDER = "dataset"
# Define dataset folder and queue folder
QUEUE_FOLDER = "queue"


class ProgressBar:
    def __init__(self, pct=0):
        self._pct = pct
        self.validate_percentage(pct)

    def validate_percentage(self, pct):
        if not (0 <= pct <= 100):
            raise ValueError("Percentage must be between 0 and 100.")
        self._pct = pct

    def progress(self, pct):
        self.validate_percentage(pct)

    def get_progress(self):
        return self._pct

    def display(self):
        bar_length = 20
        filled_length = int(bar_length * self._pct // 100)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        return f"[{bar}] {self._pct}%"
    
def generate_image(server_url, positive_prompt, character_name, resolution=(512, 512), steps=20, lora_version=None, output_path="output"):
    """
    Calls the /generate endpoint to create an image based on the given parameters and saves it locally.
    
    Parameters:
        server_url (str): The URL of the image generation server.
        positive_prompt (str): The prompt for the image generation.
        character_name (str): The character name for selecting the LORA adapter.
        resolution (tuple): The resolution of the generated image (width, height).
        steps (int): The number of steps for the image generation.
        lora_version (str): The LORA version to use (default: latest).
        output_path (str): Path to save the generated image.
    
    Returns:
        str: The seed used for generation, or None if the request fails.
    """
    # Payload for the request
    payload = {
        "positive_prompt": positive_prompt,
        "character_name": character_name,
        "resolution": list(resolution),
        "steps": steps,
        "lora_version": lora_version,
    }

    # Get the directory of the currently running script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to the config file
    output_path = os.path.join(current_dir, output_path)

    
    try:
        # Make a POST request to the server
        response = requests.post(f"{server_url}/generate", json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse the response
        result = response.json()
        image_data = result.get("image", None)
        seed = result.get("seed", None)

        if image_data and seed is not None:
            # Decode base64 image and save it
            image_base64 = image_data.split(",")[1]  # Remove the data URL prefix
            output_path = os.path.join(output_path, f"img_{seed}.png")  # Use img_{seed} for the file name
            with open(output_path, "wb") as img_file:
                img_file.write(base64.b64decode(image_base64))
            print(f"Image generated and saved as {output_path}")
        else:
            print("No image or seed returned in the response.")
        
        return output_path, seed
    
    except requests.exceptions.RequestException as e:
        print(f"Error during request: {e}")
        return None

def train_model(config_data):
    url = "http://localhost:8888/train"
    response = requests.post(url, json={"config_data": config_data})
    if response.status_code == 200:
        print(response.json()["message"])
        return True
    else:
        print(f"Error: {response.json()['detail']}")
        return False
    

def start_comfyui():
    script_path = "/media/farid/data1/projects/ComfyUI/start_comfyui.sh"

    # Check if the script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"The script {script_path} does not exist. Please create it first.")

    # Execute the shell script
    try:
        result = subprocess.run(
            [script_path],
            check=True,         # Ensures an exception is raised if the script fails
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True           # Ensures the output is a string instead of bytes
        )
        print(f"ComfyUI started successfully:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while starting ComfyUI:\n{e.stderr}")
        raise e    
    
def stop_comfyui():
    """
    Stops the ComfyUI process by executing the terminate script.
    """
    script_path = "/media/farid/data1/projects/ComfyUI/script/terminate_comfyui.sh"

    # Check if the terminate script exists
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"The script {script_path} does not exist. Please create it first.")

    # Execute the shell script
    try:
        subprocess.run(
            [script_path],
            check=True,  # Ensures an exception is raised if the script fails
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True    # Ensures the output is a string
        )
        print("ComfyUI terminated successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while stopping ComfyUI:\n{e.stderr}")
        raise e    
    
    import os
import zipfile

def create_zip_from_folder(dataset_folder, output_zip):
    """
    Create a zip file from a dataset folder.
    
    Args:
        dataset_folder (str): Path to the dataset folder.
        output_zip (str): Path for the output zip file.
    """
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"The folder '{dataset_folder}' does not exist.")
    
    # Create a zip file
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(dataset_folder):
            for file in files:
                # Write each file to the zip file
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, dataset_folder)  # Preserve folder structure in zip
                zipf.write(file_path, arcname)

    print(f"Zip file created: {output_zip}")

# Example usage
dataset_folder = "path/to/dataset_folder"  # Replace with your dataset folder path
output_zip = "path/to/output_dataset.zip"  # Replace with desired output zip file path

