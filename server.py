import chainlit as cl
import logging
import os
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to the destination folder
DESTINATION_FOLDER = "dataset"
CONFIG_PATH = "config/train_lora_24gb.yaml"

def clean_dataset_folder(folder_path=DESTINATION_FOLDER):
    """
    Removes all files from the specified folder. Creates the folder if it doesn't exist.
    """
    try:
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        os.makedirs(folder_path)
        logging.info(f"Cleaned and prepared dataset folder: {folder_path}")
    except Exception as e:
        logging.error(f"Error cleaning dataset folder: {e}")
        raise

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

@cl.on_chat_start
async def start_chat():
    await cl.Message(content="Virtual Me BOT").send()
    
    files = None
    
    # Wait for the user to upload a file
    while files == None:
        files = await cl.AskFileMessage(
            accept={"image/png": ['.png', '.jpg']},
            max_files=10,
            content="Téléverse 10 photos de toi pour commencer..."
        ).send()
    
    # Assuming `files` contains the uploaded images
    image_files = files[:10]  # Limit to 10 images
    
    # Create Elements from the uploaded files
    elements = []
    for file in image_files:
        # Create an Element from each file
        element = cl.Image(name=file.name, path=file.path, display="inline", size="small")
        elements.append(element)

    # Let the user know that the system is ready
    await cl.Message(
        content="Tes photos",
        elements=elements
    ).send()
    
    prenom = await cl.AskUserMessage(
        content="Quel est ton prénom? \nTu pourras l'utiliser ensuite pour générer des photos de toi dans toutes les situations qui te passent par la tête!!!"
    ).send()
    
    await cl.Message(content=f"Merci {prenom['output']}, l'entrainement sur tes images va débuter. Cela peut durer jusqu'à 3 heures.\n Soit patient, tu seras récompensé ;-)").send()

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)