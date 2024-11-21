import os
import json
from pathlib import Path
import shutil
import base64
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI()

def encode_image_to_base64(image_path: str) -> str:
    """
    Encode image to base64 string.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        str: Base64 encoded image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_description(image_path: str, character_name: str) -> str:
    """
    Get image description using OpenAI's GPT-4O Vision.
    
    Args:
        image_path (str): Path to the image file
        character_name (str): Name of the character
        
    Returns:
        str: Description of the image
    """
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the message for GPT-4O Vision
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"""Describe this picture for flux.dev training, on the picture it's Arielle, make the description short"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=300
    )
    
    return response.choices[0].message.content

def process_images(character_name: str, folder_path: str):
    """
    Process images in the specified folder and generate corresponding JSON and text files.
    
    Args:
        character_name (str): Name of the character
        folder_path (str): Path to the folder containing images
    """
    # Create Path object
    folder = Path(folder_path)
    
    # Ensure folder exists
    if not folder.exists():
        raise ValueError(f"Folder {folder_path} does not exist")
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg')
    image_files = [f for f in folder.iterdir() if f.suffix.lower() in image_extensions]
    
    # Sort files to ensure consistent numbering
    image_files.sort()
    
    # Process each image
    for index, image_path in enumerate(image_files, start=1):
        try:
            # Create new filename
            new_name = f"{character_name}-{index}{image_path.suffix.lower()}"
            new_path = folder / new_name
            
            # Rename image file
            shutil.move(image_path, new_path)
            
            # Get image description using GPT-4O Vision
            description = get_image_description(str(new_path), character_name)
            
            # Create JSON file
            json_data = {
                "caption": f"{character_name}, {description}"
            }
            json_path = folder / f"{character_name}-{index}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=4)
            
            # Create text file
            txt_path = folder / f"{character_name}-{index}.txt"
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(f"{character_name}, {description}")
            
            print(f"Processed image {index}: {new_name}")
            
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue

def main():
    """
    Main function to run the pipeline.
    """
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI API key not found in environment variables")
    
    # Get parameters from user
    character_name = input("Enter character name: ")
    folder_path = input("Enter folder path to scan: ")
    
    try:
        process_images(character_name, folder_path)
        print("Pipeline completed successfully!")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()