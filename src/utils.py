"""
Utility functions for the Call Center Analytics Platform
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
from dotenv import load_dotenv


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Set up logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Suppress verbose logging from third-party libraries
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('google_genai').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML configuration: {e}")


def load_environment() -> Dict[str, str]:
    """Load environment variables from .env file"""
    load_dotenv()
    
    env_vars = {
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'HUGGINGFACE_TOKEN': os.getenv('HUGGINGFACE_TOKEN'),  # Optional for public models
    }
    
    # Check for required API keys (only Gemini is required)
    if not env_vars['GEMINI_API_KEY']:
        raise EnvironmentError(
            f"Missing required environment variable: GEMINI_API_KEY. "
            f"Please check your .env file."
        )
    
    return env_vars


def save_json(data: Dict[str, Any], filepath: str, indent: int = 2) -> None:
    """Save dictionary to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(filepath: str) -> Dict[str, Any]:
    """Load dictionary from JSON file"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"JSON file not found: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_audio_files(directory: str, extensions: list = None) -> list:
    """Get list of audio files from directory"""
    if extensions is None:
        extensions = ['.wav', '.WAV', '.mp3', '.MP3', '.m4a', '.flac']
    
    audio_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                audio_files.append(os.path.join(root, file))
    
    return audio_files


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if it doesn't"""
    os.makedirs(path, exist_ok=True)


def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format duration in seconds to human-readable string"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to maximum length"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def parse_call_id_datetime(call_id: str) -> Optional[datetime]:
    """
    Extract datetime from call ID
    Expects format: YYYYMMDDHHMMSS_...
    """
    try:
        dt_str = call_id[:14]
        return datetime.strptime(dt_str, "%Y%m%d%H%M%S")
    except (ValueError, IndexError):
        return None


def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing invalid characters"""
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


class ProgressTracker:
    """Track processing progress and save checkpoints"""
    
    def __init__(self, checkpoint_file: str = "progress.json"):
        self.checkpoint_file = checkpoint_file
        self.processed_files = set()
        self.failed_files = set()
        self.load_progress()
    
    def load_progress(self) -> None:
        """Load existing progress from checkpoint file"""
        if os.path.exists(self.checkpoint_file):
            try:
                with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_files = set(data.get('processed_files', []))
                    self.failed_files = set(data.get('failed_files', []))
            except Exception as e:
                logging.warning(f"Could not load progress checkpoint: {e}")
    
    def save_progress(self) -> None:
        """Save current progress to checkpoint file"""
        data = {
            'processed_files': list(self.processed_files),
            'failed_files': list(self.failed_files),
            'timestamp': datetime.now().isoformat(),
            'total_processed': len(self.processed_files),
            'total_failed': len(self.failed_files)
        }
        
        try:
            with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"Could not save progress checkpoint: {e}")
    
    def mark_processed(self, filename: str) -> None:
        """Mark file as successfully processed"""
        self.processed_files.add(filename)
    
    def mark_failed(self, filename: str) -> None:
        """Mark file as failed"""
        self.failed_files.add(filename)
    
    def is_processed(self, filename: str) -> bool:
        """Check if file has been processed"""
        return filename in self.processed_files
    
    def is_failed(self, filename: str) -> bool:
        """Check if file has failed"""
        return filename in self.failed_files
    
    def get_stats(self) -> Dict[str, int]:
        """Get processing statistics"""
        return {
            'processed': len(self.processed_files),
            'failed': len(self.failed_files),
            'total': len(self.processed_files) + len(self.failed_files)
        }
