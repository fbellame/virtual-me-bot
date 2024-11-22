import os
import shutil
import chainlit as cl
import logging
from chainlit.input_widget import Select
from image import clean_dataset_folder, DESTINATION_FOLDER
from prepare_dataset import process_images
from utils import ProgressBar, generate_image, train_model, start_comfyui, stop_comfyui
import yaml
from lora_manager import LoraMappingManager, LoraVersion
from datetime import datetime
from typing import Optional, Dict
import chainlit as cl
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Server URL
IMAGE_SERVER_URL = os.getenv('IMAGE_SERVER_URL', "http://localhost:8886")

# Get the directory of the currently running script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the full path to the config file
output_path = os.path.join(current_dir, "config")    
config_file = os.path.join(output_path, "lora_mappings.json")

def load_users(file_path):
    """Load the configuration from a JSON file."""
    with open(file_path, "r") as file:
        return json.load(file)

# Load the configuration at the start of your application
output_path = os.path.join(current_dir, "user_config")    
user_config_file = os.path.join(output_path, "users.json")
user_config = load_users(user_config_file)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Look for the user in the configuration
    for user in user_config.get("users", []):
        if username == user["username"] and password == user["password"]:
            return cl.User(
                identifier=user["username"], 
                metadata={"role": user["role"], "provider": user["provider"]}
            )
    return None

@cl.oauth_callback
def oauth_callback(
  provider_id: str,
  token: str,
  raw_user_data: Dict[str, str],
  default_user: cl.User,
) -> Optional[cl.User]:
  return default_user

@cl.on_chat_start
async def start_chat():

    app_user = cl.user_session.get("user")
    await cl.Message(f"Salut {app_user.identifier}").send()    

    manager = LoraMappingManager(config_file)

    characters = manager.get_all_characters()

    settings = await cl.ChatSettings([Select(id="Image", label="Configuration LORA", values=characters, initial_index=1)]).send()
    await setup_agent(settings)

    response = await cl.AskActionMessage(
        content="Que voulez-vous faire?",
        actions=[
            cl.Action(name="entrainement", value="entrainement", label="Entrainement"),
            cl.Action(name="generation", value="generation",label="Generation"),
        ]
    ).send()

    # Traitement de la réponse en fonction du bouton cliqué
    if response is not None and response['value'] == "generation":

        while True:     
            response = None   

            # Dynamically create actions for each character
            character_actions = [
                cl.Action(name=character, value=character, label=character.capitalize())
                for character in characters
            ]
            character_actions.append(cl.Action(name="aucun", value="", label="Aucun"))

            # Send character selection prompt
            response = await cl.AskActionMessage(
                content="Quel personnage?",
                actions=character_actions
            ).send()

            if response is not None:
                character_name = response['value']

                perso = ""
                if character_name != '':
                    perso = f" en incluant {response["label"]} au début de ton prompt"

                response = await cl.AskUserMessage(
                    content=f"Décrit moi l'image que tu veux générer{perso}:", 
                    timeout=3000).send()

                if response is not None:
                    positive_prompt = response['output']

                    element = await generate_image_tool(positive_prompt, character_name)

                    # Let the user know that the system is ready
                    await cl.Message(
                        content="Ta photo:",
                        elements=[element]
                    ).send()

    elif response is not None and response['value'] == "entrainement":        
        files = None
        
        # Wait for the user to upload a file
        while files == None:
            files = await cl.AskFileMessage(
                accept={"image/png": ['.png', '.jpg']},
                max_files=10,
                max_size_mb=5,
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
        message = cl.Message(
            content="Tes photos",
            elements=elements
        )
        result = await message.send()
        
        prenom = await cl.AskUserMessage(
            content="Quel est ton prénom? \nTu pourras l'utiliser ensuite pour générer des photos de toi dans toutes les situations qui te passent par la tête!!!"
        ).send()

        result = await prepare_image_tool(message, prenom['output'])
        print(result)
        await cl.Message(content="Dataset réalisé...").send()        
    
        await cl.Message(content=f"Merci {prenom['output']}, l'entrainement sur tes images va débuter. Cela peut durer jusqu'à 3 heures.\n Soit patient, tu seras récompensé ;-)").send()

        result = await train_image_tool(message, prenom['output'])

        if result:
            await cl.Message(content="Entrainement terminé...").send()
        else:
            await cl.Message(content="Désolé une erreur s'est produite...").send()

@cl.step(type="tool")
async def generate_image_tool(positive_prompt: str, character_name : str):

    img, seed = generate_image(
        server_url=IMAGE_SERVER_URL,
        positive_prompt=positive_prompt,
        character_name=character_name,
        resolution=(1024, 1024),
        steps=20
    )

    element = cl.Image(name=f"img_{seed}.png", path=img, display="inline", size="large")

    return element

@cl.on_settings_update
async def setup_agent(settings):
    logging.info("Setup agent with following settings: ", settings)
    cl.user_session.set("image_model", settings["Image"])

if __name__ == "__main__":
    from chainlit.cli import run_chainlit
    run_chainlit(__file__)

@cl.step(type="tool")
async def prepare_image_tool(message: cl.Message, character_name : str):
    pg = ProgressBar(0)

    # Clean the dataset folder before processing new images
    clean_dataset_folder()

    input_images = message.elements
    for image in input_images:
        if image.type == 'image':
            # Destination path for the copied image
            destination_path = os.path.join(DESTINATION_FOLDER, image.name)
            # Copy the image
            shutil.copy(image.path, destination_path)

    process_images(character_name=character_name, folder_path=DESTINATION_FOLDER)    

    return True   

@cl.step(type="tool")
async def train_image_tool(message: cl.Message, character_name : str):

    # Call the stop_comfyui function
    try:
        stop_comfyui()
    except Exception as e:
        print(f"Error stopping ComfyUI: {e}")
        return False

    # Load the config file
    with open("config/train_lora_24gb.yaml", "r") as f:
        config_data = yaml.safe_load(f)

    # Update the configuration with the character name
    config_data['config']['name'] = f"{character_name}_flux_lora_v6"
    config_data['config']['process'][0]['training_folder'] = f"output_{character_name}"
    config_data['config']['process'][0]['trigger_word'] = character_name
    config_data['meta']['name'] = f"[{character_name}]"

    # Debug: Print the updated configuration for verification
    print(config_data)

    if train_model(config_data):

        manager = LoraMappingManager(config_file)
        
        # Add a mapping
        new_version = LoraVersion(
            lora_name=f"{config_data['config']['name']}.safetensors",
            version="6.0.0",
            created_at=datetime.now().isoformat(),
            description="Initial training"
        )
        manager.add_mapping(character_name, new_version)

        # move Lora adapter into ComfyUI  
        shutil.move(
            f"/media/farid/data1/projects/ai-toolkit/{config_data['config']['process'][0]['training_folder']}/{config_data['config']['name']}/{config_data['config']['name']}.safetensors", 
            "/media/farid/data1/projects/ComfyUI/models")

        return True
    
    # Call the stop_comfyui function
    try:
        start_comfyui()
    except Exception as e:
        print(f"Error stopping ComfyUI: {e}")
        return False    
    
    return False
