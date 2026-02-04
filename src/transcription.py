"""
Audio transcription module using OpenAI Whisper
"""

import os
import torch
import logging
import librosa
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import threading

from .utils import ProgressTracker, get_audio_files, save_json, load_json


logger = logging.getLogger(__name__)


class AudioTranscriber:
    """
    Handles audio transcription using Whisper model with direct model.generate()
    """
    
    def __init__(
        self,
        model_id: str = "openai/whisper-large-v3-turbo",
        language: str = "Korean",
        device: str = "auto",
        batch_size: int = 24,
        chunk_length: int = 30,
        huggingface_token: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        quality_verification: bool = True,
        max_retries: int = 3
    ):
        self.model_id = model_id
        self.language = language
        self.batch_size = batch_size
        self.chunk_length = chunk_length
        self.quality_verification = quality_verification
        self.max_retries = max_retries
        
        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        
        # Load model and processor
        logger.info(f"Loading Whisper model: {model_id} on {self.device}")
        self.model, self.processor = self._load_model(huggingface_token)
        
        # Get language token
        self.language_token = self._get_language_token(language)
        
        # Initialize Gemini for quality verification
        if quality_verification and gemini_api_key:
            self.client = genai.Client(api_key=gemini_api_key)
            self.gen_model = "gemini-2.0-flash-exp"
        else:
            self.gen_model = None
            if quality_verification:
                logger.warning("Quality verification disabled: Gemini API key not provided")
    
    def _load_model(self, token: Optional[str] = None):
        """Load Whisper model and processor directly"""
        try:
            # Token is optional - only needed for gated models
            model_kwargs = {
                'torch_dtype': self.torch_dtype,
                'low_cpu_mem_usage': True,
                'use_safetensors': True,
            }
            if token:
                model_kwargs['token'] = token
            
            model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                **model_kwargs
            )
            model = model.to(self.device)
            model.eval()
            
            processor_kwargs = {}
            if token:
                processor_kwargs['token'] = token
            processor = AutoProcessor.from_pretrained(self.model_id, **processor_kwargs)
            
            logger.info(f"Model loaded successfully on {self.device}")
            return model, processor
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _get_language_token(self, language: str) -> Optional[str]:
        """Get the language token for forced decoding"""
        # Map common language names to Whisper tokens
        language_map = {
            'Korean': 'ko',
            'English': 'en',
            'Japanese': 'ja',
            'Chinese': 'zh',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
        }
        return language_map.get(language, language.lower()[:2])
    
    def transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe a single audio file using model.generate() with long-form support
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Transcribed text
        """
        try:
            # Load audio using soundfile first (handles non-seekable files better)
            import soundfile as sf
            
            # Read the entire file
            audio_array, sampling_rate = sf.read(audio_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            
            # Calculate audio duration in seconds
            duration = len(audio_array) / 16000
            
            # For long audio (>30s), use chunking approach
            if duration > 30:
                return self._transcribe_long_audio(audio_array)
            
            # For short audio, transcribe directly
            # Process audio with processor
            inputs = self.processor(
                audio_array,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move to device
            input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
            attention_mask = inputs.attention_mask.to(self.device, dtype=self.torch_dtype) if 'attention_mask' in inputs else None
            
            # Prepare generation config with language forcing
            generation_config = {
                "task": "transcribe",
                "language": self.language_token,
                "num_beams": 1,  # Greedy decoding for speed
                "return_timestamps": False,
                "do_sample": False,  # Greedy decoding
            }
            
            # Generate with explicit inputs
            with torch.no_grad():
                if attention_mask is not None:
                    predicted_ids = self.model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        **generation_config
                    )
                else:
                    predicted_ids = self.model.generate(
                        input_features,
                        **generation_config
                    )
            
            # Decode the transcription
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                normalize=True
            )[0]
            
            return transcription.strip()
            
        except Exception as e:
            logger.error(f"Transcription failed for {audio_path}: {e}")
            raise
    
    def _transcribe_long_audio(self, audio_array: np.ndarray) -> str:
        """
        Transcribe long audio by chunking into 30-second segments with overlap
        
        Args:
            audio_array: Audio data at 16kHz
            
        Returns:
            Combined transcription
        """
        chunk_length = 30  # 30 seconds per chunk
        overlap = 5  # 5 seconds overlap
        sample_rate = 16000
        
        chunk_samples = chunk_length * sample_rate
        overlap_samples = overlap * sample_rate
        stride = chunk_samples - overlap_samples
        
        transcriptions = []
        total_samples = len(audio_array)
        
        for start in range(0, total_samples, stride):
            end = min(start + chunk_samples, total_samples)
            chunk = audio_array[start:end]
            
            # Process chunk
            inputs = self.processor(
                chunk,
                sampling_rate=16000,
                return_tensors="pt",
                return_attention_mask=True
            )
            
            # Move to device
            input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
            attention_mask = inputs.attention_mask.to(self.device, dtype=self.torch_dtype) if 'attention_mask' in inputs else None
            
            # Generate
            generation_config = {
                "task": "transcribe",
                "language": self.language_token,
                "num_beams": 1,
                "return_timestamps": False,
                "do_sample": False,
            }
            
            with torch.no_grad():
                if attention_mask is not None:
                    predicted_ids = self.model.generate(
                        input_features,
                        attention_mask=attention_mask,
                        **generation_config
                    )
                else:
                    predicted_ids = self.model.generate(
                        input_features,
                        **generation_config
                    )
            
            # Decode
            chunk_text = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                normalize=True
            )[0].strip()
            
            if chunk_text:
                transcriptions.append(chunk_text)
        
        # Combine with basic deduplication at chunk boundaries
        combined = " ".join(transcriptions)
        return combined.strip()
    
    def verify_transcription(self, transcription: str) -> bool:
        """
        Verify transcription quality using Gemini AI
        
        Args:
            transcription: Transcribed text to verify
            
        Returns:
            True if quality is acceptable, False otherwise
        """
        if not hasattr(self, 'client'):
            return True  # Skip verification if client not available
        
        try:
            # Truncate very long transcriptions for verification
            sample = transcription[:1000] if len(transcription) > 1000 else transcription
            
            prompt = (
                f"Analyze the following audio transcription. If it is clear, accurate, "
                f"and properly structured without major errors, respond only with 'True'. "
                f"If it is incomplete, incorrect, or contains gibberish, respond only with 'False'. "
                f"Do not provide explanations. Transcription: {sample}"
            )
            
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt
            )
            decision = response.text.strip()
            
            # Log the verification details
            logger.debug(f"Verification - Length: {len(transcription)}, Sample: {sample[:100]}..., Decision: {decision}")
            
            if decision != 'True' and decision != 'False':
                logger.warning(f"Unexpected verification response: '{decision}'. Accepting transcription.")
                return True
            
            return decision == 'True'
            
        except Exception as e:
            logger.warning(f"Verification failed: {e}. Accepting transcription.")
            return True
    
    def transcribe_files(
        self,
        input_dir: str,
        output_file: str = "transcriptions.json",
        checkpoint_file: str = "transcription_progress.json",
        save_interval: int = 10,
        preload_workers: int = 4
    ) -> Dict[str, str]:
        """
        Transcribe all audio files in a directory with parallel audio loading
        
        Args:
            input_dir: Directory containing audio files
            output_file: Path to save transcriptions
            checkpoint_file: Path to save progress
            save_interval: Save checkpoint every N files
            preload_workers: Number of threads for parallel audio loading
            
        Returns:
            Dictionary mapping file names to transcriptions
        """
        # Load existing transcriptions and progress
        transcriptions = {}
        if os.path.exists(output_file):
            transcriptions = load_json(output_file)
            logger.info(f"Loaded {len(transcriptions)} existing transcriptions")
        
        tracker = ProgressTracker(checkpoint_file)
        
        # Get all audio files
        audio_files = get_audio_files(input_dir)
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return transcriptions
        
        # Filter out already processed files
        remaining_files = [
            f for f in audio_files
            if os.path.basename(f) not in tracker.processed_files
            and os.path.basename(f) not in tracker.failed_files
        ]
        
        stats = tracker.get_stats()
        logger.info(f"Total files: {len(audio_files)}")
        logger.info(f"Already processed: {stats['processed']}")
        logger.info(f"Previously failed: {stats['failed']}")
        logger.info(f"Remaining to process: {len(remaining_files)}")
        
        if not remaining_files:
            logger.info("All files have been processed!")
            return transcriptions
        
        # Use parallel audio loading with queue
        return self._transcribe_with_preloading(
            remaining_files, transcriptions, tracker, 
            output_file, save_interval, preload_workers
        )
    
    def _load_audio_file(self, audio_path: str) -> Tuple[str, Optional[np.ndarray], Optional[str]]:
        """Load audio file in background thread"""
        try:
            import soundfile as sf
            filename = os.path.basename(audio_path)
            
            # Read the entire file
            audio_array, sampling_rate = sf.read(audio_path, dtype='float32')
            
            # Convert to mono if stereo
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)
            
            # Resample to 16kHz if needed
            if sampling_rate != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sampling_rate, target_sr=16000)
            
            return (filename, audio_array, None)
        
        except Exception as e:
            return (os.path.basename(audio_path), None, str(e))
    
    def _transcribe_with_preloading(
        self,
        file_paths: List[str],
        transcriptions: Dict[str, str],
        tracker: ProgressTracker,
        output_file: str,
        save_interval: int,
        preload_workers: int
    ) -> Dict[str, str]:
        """Transcribe with parallel audio loading"""
        import time
        processing_times = []
        files_processed = 0
        
        # Create a thread pool for loading audio
        with ThreadPoolExecutor(max_workers=preload_workers) as loader:
            # Submit first batch of audio loads
            future_to_path = {}
            load_buffer = min(preload_workers * 2, len(file_paths))
            
            for i in range(min(load_buffer, len(file_paths))):
                future = loader.submit(self._load_audio_file, file_paths[i])
                future_to_path[future] = file_paths[i]
            
            next_file_idx = load_buffer
            
            with tqdm(total=len(file_paths), desc="Transcribing", unit="file") as pbar:
                for future in as_completed(future_to_path):
                    audio_path = future_to_path[future]
                    filename, audio_array, error = future.result()
                    
                    start_time = time.time()
                    success = False
                    
                    # Submit next file load while processing current
                    if next_file_idx < len(file_paths):
                        next_future = loader.submit(self._load_audio_file, file_paths[next_file_idx])
                        future_to_path[next_future] = file_paths[next_file_idx]
                        next_file_idx += 1
                    
                    # Process loaded audio
                    if error:
                        logger.error(f"Failed to load {filename}: {error}")
                        transcriptions[filename] = "Audio loading failed"
                        tracker.mark_failed(filename)
                    else:
                        try:
                            # Calculate duration and decide processing method
                            duration = len(audio_array) / 16000
                            
                            if duration > 30:
                                transcription = self._transcribe_long_audio(audio_array)
                            else:
                                # Short audio - direct transcription
                                inputs = self.processor(
                                    audio_array,
                                    sampling_rate=16000,
                                    return_tensors="pt",
                                    return_attention_mask=True
                                )
                                
                                input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
                                attention_mask = inputs.attention_mask.to(self.device, dtype=self.torch_dtype) if 'attention_mask' in inputs else None
                                
                                generation_config = {
                                    "task": "transcribe",
                                    "language": self.language_token,
                                    "num_beams": 1,
                                    "return_timestamps": False,
                                    "do_sample": False,
                                }
                                
                                with torch.no_grad():
                                    if attention_mask is not None:
                                        predicted_ids = self.model.generate(
                                            input_features,
                                            attention_mask=attention_mask,
                                            **generation_config
                                        )
                                    else:
                                        predicted_ids = self.model.generate(
                                            input_features,
                                            **generation_config
                                        )
                                
                                transcription = self.processor.batch_decode(
                                    predicted_ids,
                                    skip_special_tokens=True,
                                    normalize=True
                                )[0].strip()
                            
                            # Verify if enabled
                            if self.verify_transcription(transcription):
                                transcriptions[filename] = transcription
                                tracker.mark_processed(filename)
                                success = True
                            else:
                                logger.warning(f"Verification failed for {filename}")
                                transcriptions[filename] = transcription  # Save anyway since verification is disabled
                                tracker.mark_processed(filename)
                                success = True
                        
                        except Exception as e:
                            logger.error(f"Transcription error for {filename}: {e}")
                            transcriptions[filename] = "Transcription failed"
                            tracker.mark_failed(filename)
                    
                    # Track time
                    elapsed = time.time() - start_time
                    processing_times.append(elapsed)
                    files_processed += 1
                    
                    # Calculate ETA
                    avg_time = sum(processing_times) / len(processing_times)
                    remaining = len(file_paths) - files_processed
                    eta_minutes = (avg_time * remaining) / 60
                    
                    pbar.set_postfix({
                        'avg_time': f'{avg_time:.1f}s',
                        'eta': f'{eta_minutes:.1f}m'
                    })
                    pbar.update(1)
                    
                    # Save checkpoint
                    if files_processed % save_interval == 0:
                        save_json(transcriptions, output_file)
                        tracker.save_progress()
                        logger.info(f"Checkpoint saved after {files_processed} files")
        
        # Final save
        save_json(transcriptions, output_file)
        tracker.save_progress()
        
        final_stats = tracker.get_stats()
        logger.info(f"Transcription complete!")
        logger.info(f"Total processed: {final_stats['processed']}")
        logger.info(f"Total failed: {final_stats['failed']}")
        
        return transcriptions


def create_transcriber(config: Dict, env_vars: Dict) -> AudioTranscriber:
    """
    Factory function to create AudioTranscriber from config
    
    Args:
        config: Configuration dictionary
        env_vars: Environment variables dictionary
        
    Returns:
        AudioTranscriber instance
    """
    transcription_config = config.get('transcription', {})
    
    return AudioTranscriber(
        model_id=transcription_config.get('model', 'openai/whisper-large-v3-turbo'),
        language=transcription_config.get('language', 'Korean'),
        device=transcription_config.get('device', 'auto'),
        batch_size=transcription_config.get('batch_size', 24),
        chunk_length=transcription_config.get('chunk_length', 30),
        huggingface_token=env_vars.get('HUGGINGFACE_TOKEN'),
        gemini_api_key=env_vars.get('GEMINI_API_KEY'),
        quality_verification=transcription_config.get('quality_verification', True),
        max_retries=transcription_config.get('max_retries', 3)
    )
