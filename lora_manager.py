from typing import Dict, List, Optional, Any
import json
import os
from datetime import datetime
import logging
from pathlib import Path
import shutil
from pydantic import BaseModel, field_validator
import semantic_version

# Get the directory of the currently running script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the config file
config_path = os.path.join(script_dir, "config", "config.json")

print(f"Config path: {config_path}")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LoraVersion(BaseModel):
    lora_name: str
    version: str
    created_at: str
    description: Optional[str] = None
    training_config: Optional[Dict[str, Any]] = None
    file_path: Optional[str] = None

    @field_validator('version')
    def validate_version(cls, v):
        try:
            semantic_version.Version(v)
            return v
        except ValueError:
            raise ValueError("Version must follow semantic versioning (e.g., 1.0.0)")

class LoraMappingManager:
    def __init__(self, mapping_file: str, backup_dir: str = "./config/backups"):
        """
        Initialize the LORA mapping manager.
        
        Args:
            mapping_file: Path to the JSON file storing the mappings
            backup_dir: Directory for storing mapping file backups
        """
        self.mapping_file = mapping_file
        self.backup_dir = backup_dir
        self.mappings: Dict[str, List[LoraVersion]] = {}
        self._ensure_dirs()
        self._load_mappings()

    def _ensure_dirs(self) -> None:
        """Ensure required directories exist."""
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        Path(os.path.dirname(self.mapping_file)).mkdir(parents=True, exist_ok=True)

    def _create_backup(self) -> None:
        """Create a backup of the current mapping file."""
        if os.path.exists(self.mapping_file):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.backup_dir, f"lora_mappings_{timestamp}.json")
            shutil.copy2(self.mapping_file, backup_path)
            
            # Keep only last 5 backups
            backups = sorted(Path(self.backup_dir).glob("lora_mappings_*.json"))
            for backup in backups[:-5]:
                backup.unlink()

    def _load_mappings(self) -> None:
        """Load mappings from file, creating if necessary."""
        try:
            if os.path.exists(self.mapping_file):
                with open(self.mapping_file, 'r') as f:
                    data = json.load(f)
                    # Convert stored data to LoraVersion objects
                    self.mappings = {
                        character: [LoraVersion(**version) for version in versions]
                        for character, versions in data.items()
                    }
            else:
                self.mappings = {}
                self._save_mappings()
        except Exception as e:
            logger.error(f"Error loading mappings: {e}")
            raise

    def _save_mappings(self) -> None:
        """Save current mappings to file with backup."""
        try:
            self._create_backup()
            with open(self.mapping_file, 'w') as f:
                # Convert LoraVersion objects to dict for JSON serialization
                data = {
                    character: [version.dict() for version in versions]
                    for character, versions in self.mappings.items()
                }
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving mappings: {e}")
            raise

    def add_mapping(self, character_name: str, lora_version: LoraVersion) -> None:
        """
        Add a new LORA mapping for a character.
        
        Args:
            character_name: Name of the character
            lora_version: LoraVersion object containing the mapping details
        """
        try:
            character_name = character_name.lower()
            if character_name not in self.mappings:
                self.mappings[character_name] = []

            # Check for version conflict
            for existing in self.mappings[character_name]:
                if existing.version == lora_version.version:
                    raise ValueError(f"Version {lora_version.version} already exists for {character_name}")

            # Add new version
            self.mappings[character_name].append(lora_version)
            
            # Sort versions using semantic versioning
            self.mappings[character_name].sort(
                key=lambda x: semantic_version.Version(x.version),
                reverse=True
            )
            
            self._save_mappings()
            logger.info(f"Added new LORA mapping for {character_name} version {lora_version.version}")
        except Exception as e:
            logger.error(f"Error adding mapping: {e}")
            raise

    def get_latest_lora(self, character_name: str) -> Optional[str]:
        """
        Get the latest LORA adapter name for a character.
        
        Args:
            character_name: Name of the character
        
        Returns:
            The LORA adapter name or None if not found
        """
        character_name = character_name.lower()
        if character_name in self.mappings and self.mappings[character_name]:
            return self.mappings[character_name][0].lora_name
        return None

    def get_specific_version(self, character_name: str, version: str) -> Optional[str]:
        """
        Get a specific LORA version for a character.
        
        Args:
            character_name: Name of the character
            version: Version string
        
        Returns:
            The LORA adapter name or None if not found
        """
        character_name = character_name.lower()
        if character_name in self.mappings:
            for mapping in self.mappings[character_name]:
                if mapping.version == version:
                    return mapping.lora_name
        return None

    def get_all_versions(self, character_name: str) -> List[LoraVersion]:
        """
        Get all LORA versions for a character.
        
        Args:
            character_name: Name of the character
        
        Returns:
            List of LoraVersion objects
        """
        character_name = character_name.lower()
        return self.mappings.get(character_name, [])

    def get_all_characters(self) -> List[str]:
        """Get all character names with LORA mappings."""
        return list(self.mappings.keys())

    def update_mapping(self, character_name: str, version: str, updates: Dict[str, Any]) -> None:
        """
        Update an existing LORA mapping.
        
        Args:
            character_name: Name of the character
            version: Version to update
            updates: Dictionary of fields to update
        """
        try:
            character_name = character_name.lower()
            if character_name not in self.mappings:
                raise ValueError(f"Character {character_name} not found")

            for i, mapping in enumerate(self.mappings[character_name]):
                if mapping.version == version:
                    updated_mapping = mapping.copy(update=updates)
                    self.mappings[character_name][i] = updated_mapping
                    self._save_mappings()
                    logger.info(f"Updated LORA mapping for {character_name} version {version}")
                    return

            raise ValueError(f"Version {version} not found for {character_name}")
        except Exception as e:
            logger.error(f"Error updating mapping: {e}")
            raise

    def remove_mapping(self, character_name: str, version: str) -> None:
        """
        Remove a specific LORA mapping.
        
        Args:
            character_name: Name of the character
            version: Version to remove
        """
        try:
            character_name = character_name.lower()
            if character_name not in self.mappings:
                raise ValueError(f"Character {character_name} not found")

            initial_length = len(self.mappings[character_name])
            self.mappings[character_name] = [
                m for m in self.mappings[character_name] if m.version != version
            ]

            if len(self.mappings[character_name]) == initial_length:
                raise ValueError(f"Version {version} not found for {character_name}")

            if not self.mappings[character_name]:
                del self.mappings[character_name]

            self._save_mappings()
            logger.info(f"Removed LORA mapping for {character_name} version {version}")
        except Exception as e:
            logger.error(f"Error removing mapping: {e}")
            raise

    def search_mappings(self, query: str) -> Dict[str, List[LoraVersion]]:
        """
        Search for characters or LORA names matching a query.
        
        Args:
            query: Search string
        
        Returns:
            Dictionary of matching character names and their versions
        """
        query = query.lower()
        results = {}
        
        for character, versions in self.mappings.items():
            if query in character.lower():
                results[character] = versions
            else:
                matching_versions = [
                    v for v in versions 
                    if query in v.lora_name.lower() or 
                    (v.description and query in v.description.lower())
                ]
                if matching_versions:
                    results[character] = matching_versions
                    
        return results

    def validate_file_exists(self, lora_name: str, lora_dir: str) -> bool:
        """
        Validate that a LORA file exists in the specified directory.
        
        Args:
            lora_name: Name of the LORA file
            lora_dir: Directory to check
        
        Returns:
            True if file exists, False otherwise
        """
        return os.path.exists(os.path.join(lora_dir, lora_name))

# Example usage
if __name__ == "__main__":
    # Initialize manager
    manager = LoraMappingManager("./config/lora_mappings.json")
    
    # Add a mapping
    new_version = LoraVersion(
        lora_name="alice_v1.safetensors",
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        description="Initial training"
    )
    manager.add_mapping("alice", new_version)
    
    # Get latest version
    latest = manager.get_latest_lora("alice")
    print(f"Latest LORA for Alice: {latest}")
    
    # Search mappings
    results = manager.search_mappings("alice")
    print(f"Search results: {results}")