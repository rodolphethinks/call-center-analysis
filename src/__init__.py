"""
Package initialization
"""

from .utils import (
    setup_logging,
    load_config,
    load_environment,
    save_json,
    load_json,
    get_audio_files,
    ensure_directory,
    ProgressTracker
)

from .transcription import AudioTranscriber, create_transcriber
from .analysis import ConversationAnalyzer, create_analyzer
from .database import CallCenterDatabase, create_database


__version__ = "1.0.0"

__all__ = [
    'setup_logging',
    'load_config',
    'load_environment',
    'save_json',
    'load_json',
    'get_audio_files',
    'ensure_directory',
    'ProgressTracker',
    'AudioTranscriber',
    'create_transcriber',
    'ConversationAnalyzer',
    'create_analyzer',
    'CallCenterDatabase',
    'create_database',
]
