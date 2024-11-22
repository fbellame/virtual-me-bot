from fastapi import FastAPI, File, HTTPException, Path as FastAPIPath
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import uuid
import os
from dotenv import load_dotenv
import logging
import requests
from typing import Dict, Any, Optional
from pydantic import BaseModel
from datetime import datetime
from pathlib import Path
from enum import Enum
from collections import deque
import threading
from lora_manager import LoraMappingManager
from contextlib import asynccontextmanager
from image import generate_images
from utils import ProgressBar
from fastapi import HTTPException
import io
from PIL import Image
import base64

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Server configuration
COMFYUI_SERVER = os.getenv('COMFYUI_SERVER_ADDRESS', 'localhost:8188')
AI_TOOLKIT_SERVER = os.getenv('AI_TOOLKIT_SERVER', 'http://localhost:8888')
WORKFLOW_PATH = "./workflow/workflow.json"
LORA_MAPPINGS_PATH = "./config/lora_mappings.json"
QUEUE_CHECK_INTERVAL = 30  # seconds

# Ensure config directory exists
Path("./config").mkdir(exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application"""
    # Startup
    logger.info("Starting up training queue manager...")
    await queue_manager.start()
    
    yield
    
    # Shutdown
    logger.info("Shutting down training queue manager...")
    await queue_manager.stop()

app = FastAPI(
    title="Image Generation and Training Proxy Server",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Training status enum
class TrainingStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Pydantic models
class GenerateImageRequest(BaseModel):
    positive_prompt: str
    character_name: str
    resolution: tuple[int, int] = (512, 512)
    steps: int = 25
    lora_version: Optional[str] = None

class TrainingConfig(BaseModel):
    config_data: Dict[str, Any]
    character_name: str
    description: Optional[str] = None

class LoraMapping(BaseModel):
    character_name: str
    lora_name: str
    version: str
    created_at: str
    description: Optional[str] = None

class TrainingRequest(BaseModel):
    id: str
    character_name: str
    config: TrainingConfig
    status: TrainingStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None

# Training Queue Manager
class TrainingQueueManager:
    def __init__(self):
        self.queue: deque[TrainingRequest] = deque()
        self.current_training: Optional[TrainingRequest] = None
        self.training_history: Dict[str, TrainingRequest] = {}
        self._lock = threading.Lock()
        self.queue_task = None

    async def start(self):
        """Initialize the queue processor when the application starts"""
        if self.queue_task is None:
            self.queue_task = asyncio.create_task(self.process_queue())
            logger.info("Training queue processor started")

    async def stop(self):
        """Clean up when the application stops"""
        if self.queue_task is not None:
            self.queue_task.cancel()
            try:
                await self.queue_task
            except asyncio.CancelledError:
                pass
            self.queue_task = None
            logger.info("Training queue processor stopped")

    def add_to_queue(self, config: TrainingConfig) -> str:
        request_id = str(uuid.uuid4())
        training_request = TrainingRequest(
            id=request_id,
            character_name=config.character_name,
            config=config,
            status=TrainingStatus.QUEUED,
            created_at=datetime.now()
        )

        with self._lock:
            self.queue.append(training_request)
            self.training_history[request_id] = training_request

        logger.info(f"Added training request {request_id} to queue for character {config.character_name}")
        return request_id

    async def check_ai_toolkit_status(self) -> bool:
        try:
            response = requests.get(f"{AI_TOOLKIT_SERVER}/status")
            return response.json().get("status") != "busy"
        except Exception as e:
            logger.error(f"Error checking AI-Toolkit status: {e}")
            return False

    async def process_queue(self):
        while True:
            try:
                if not self.current_training and self.queue:
                    if await self.check_ai_toolkit_status():
                        with self._lock:
                            training_request = self.queue.popleft()
                            self.current_training = training_request
                            training_request.status = TrainingStatus.RUNNING
                            training_request.started_at = datetime.now()

                        try:
                            # Start training
                            response = requests.post(
                                f"{AI_TOOLKIT_SERVER}/train",
                                json=training_request.config.dict()
                            )
                            response.raise_for_status()

                            # Wait for training completion
                            while not await self.check_ai_toolkit_status():
                                await asyncio.sleep(QUEUE_CHECK_INTERVAL)

                            # Update training status
                            training_request.status = TrainingStatus.COMPLETED
                            training_request.completed_at = datetime.now()

                        except Exception as e:
                            training_request.status = TrainingStatus.FAILED
                            training_request.error_message = str(e)
                            logger.error(f"Training failed for request {training_request.id}: {e}")

                        finally:
                            self.current_training = None

                await asyncio.sleep(QUEUE_CHECK_INTERVAL)
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(QUEUE_CHECK_INTERVAL)

    def get_queue_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "current_training": self.current_training.dict() if self.current_training else None,
                "queue_length": len(self.queue),
                "queued_requests": [req.dict() for req in self.queue]
            }

    def get_training_status(self, request_id: str) -> Optional[TrainingRequest]:
        return self.training_history.get(request_id)

# Initialize managers
lora_manager = LoraMappingManager(mapping_file="./config/lora_mappings.json")
queue_manager = TrainingQueueManager()

# API endpoints
@app.post("/generate")
async def generate_image(request: GenerateImageRequest):
    try:
        lora_name = None
        # Get appropriate LORA adapter
        if request.character_name != "":
            if request.lora_version:
                lora_name = lora_manager.get_specific_version(
                    request.character_name,
                    request.lora_version
                )
            else:
                lora_name = lora_manager.get_latest_lora(request.character_name)
                
            if not lora_name:
                raise HTTPException(
                    status_code=404,
                    detail=f"No LORA adapter found for character: {request.character_name}"
                )
            
            print(f"Using Lora adapter {lora_name}")
        
        pg =ProgressBar()
        images, seed = generate_images(
            positive_prompt=request.positive_prompt, 
            lora_adapter=lora_name,
            progress_bar=pg, 
            resolution=request.resolution)    

        # Convert images to base64 encoded strings
        base64_images = []

        for key, image_data in images.items():
            try:
                # Handle list of bytes; assume first element is the image
                if isinstance(image_data, list) and len(image_data) > 0:
                    image_data = image_data[0]  # Extract the first item from the list

                # Decode the image data into a PIL Image object
                image = Image.open(io.BytesIO(image_data))

                # Ensure image is in the correct mode
                if image.mode != 'RGB':
                    image = image.convert('RGB')

                # Create a byte buffer
                buffered = io.BytesIO()

                # Save the image to the buffer
                image.save(buffered, format="PNG")

                # Reset buffer pointer to the beginning
                buffered.seek(0)

                # Encode to base64
                base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
                base64_images.append(base64_image)
            except Exception as e:
                logger.error(f"Error converting image with key {key} to base64: {e}")

        # Return the images along with the seed
        return {
            "image": f"data:image/png;base64,{base64_images}",
            "seed": seed
        }            

    except Exception as e:
        logger.error(f"Error in generate_image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train_old")
async def train_model(config: TrainingConfig):
    try:
        request_id = queue_manager.add_to_queue(config)
        return JSONResponse(
            status_code=202,
            content={
                "message": "Training request queued successfully",
                "request_id": request_id
            }
        )
    except Exception as e:
        logger.error(f"Error queuing training request: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue training request")

@app.get("/train/status/{request_id}")
async def get_training_status(request_id: str):
    status = queue_manager.get_training_status(request_id)
    if not status:
        raise HTTPException(status_code=404, detail="Training request not found")
    return status

@app.get("/train/queue")
async def get_queue_status():
    return queue_manager.get_queue_status()

@app.get("/train/history")
async def get_training_history():
    return list(queue_manager.training_history.values())

@app.post("/lora-mappings")
async def add_lora_mapping(mapping: LoraMapping):
    try:
        lora_manager.add_mapping(mapping)
        return {"message": "Mapping added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lora-mappings/{character_name}")
async def get_character_loras(
    character_name: str = FastAPIPath(..., description="Name of the character")
):
    versions = lora_manager.get_all_versions(character_name)
    if not versions:
        raise HTTPException(
            status_code=404,
            detail=f"No LORA adapters found for character: {character_name}"
        )
    return versions

@app.get("/lora-mappings")
async def get_all_characters():
    return lora_manager.get_all_characters()

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/train")
async def train_endpoint(
    dataset_zip: str, yaml_path: str, username: str, character_name: str
):
    """Queue a training request"""
    import uuid
    from datetime import datetime
    import os
    import json

    # Define the queue folder
    queue_folder = "queue"
    if not os.path.exists(queue_folder):
        os.makedirs(queue_folder)

    try:
        # Create a unique ID for the request
        request_id = str(uuid.uuid4())
        
        # Prepare the request data
        submit_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_data = {
            "id": request_id,
            "dataset_zip": dataset_zip,
            "yaml_path": yaml_path,
            "submit_time": submit_time,
            "username": username,
            "character_name": character_name,
            "status": "queued"
        }
        
        # Define the output file path
        json_path = os.path.join(queue_folder, f"{request_id}.json")
        
        # Save the request to the queue folder
        with open(json_path, "w") as f:
            json.dump(request_data, f, indent=4)
        
        return JSONResponse(
            status_code=200,
            content={"message": "Training request queued successfully.", "request_id": request_id}
        )
    except Exception as e:
        logger.error(f"Error queuing training request: {e}")
        raise HTTPException(status_code=500, detail="Failed to queue training request.")
    

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8886)