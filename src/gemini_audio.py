"""
Direct audio analysis using Gemini 2.5 Flash Lite
"""

import os
import logging
import signal
import sys
from typing import Dict, Optional, List, Literal
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from google import genai
import json
from pydantic import BaseModel, Field

# Add config directory to path
config_dir = Path(__file__).parent.parent / 'config'
sys.path.insert(0, str(config_dir))

try:
    from prompt import KEYWORDS as ISSUE_CATEGORIZATION_PROMPT
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("Could not import issue categorization prompt from config/prompt.py")
    ISSUE_CATEGORIZATION_PROMPT = ""

from .utils import ProgressTracker, get_audio_files, save_json, load_json


logger = logging.getLogger(__name__)


# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    global shutdown_requested
    if not shutdown_requested:
        logger.info("\n\nShutdown requested (Ctrl+C). Finishing current tasks and saving progress...")
        shutdown_requested = True
    else:
        logger.warning("\n\nForced shutdown (Ctrl+C again). Exiting immediately...")
        sys.exit(1)

# Register signal handler
signal.signal(signal.SIGINT, signal_handler)


# Pydantic model for structured output
class CallAnalysis(BaseModel):
    customer_summary: str = Field(description="Briefly summarize the customer's statements in 1-2 sentences.")
    agent_summary: str = Field(description="Briefly summarize the agent's statements in 1-2 sentences.")
    customer_sentiment_trajectory: Literal[
        "consistently neutral",
        "consistently positive",
        "consistently negative",
        "negative to positive"
    ] = Field(description="Classify the customer's emotional trajectory. ONLY use one of the four predefined options.")
    customer_sentiment_examples: str = Field(description="Provide specific examples from the conversation that demonstrate this emotional path, based on tone of voice, word choice, and reaction to resolution attempts.")
    key_issues: List[str] = Field(description="List of categorized issues. Use EXACT category names from the predefined list. Multiple categories allowed if applicable.")
    agent_performance_overall: str = Field(description="Assess the agent's overall effectiveness in handling the call (e.g., professionalism, responsiveness, clarity).")
    agent_understanding: str = Field(description="Specifically evaluate how well the advisor understood the customer's vehicle concerns. Provide concrete examples from the conversation that demonstrate the level of understanding.")
    resolution_status: Literal[
        "Resolved",
        "Partially Resolved",
        "Escalate to next step",
        "Unresolved"
    ] = Field(description="State if the issue was resolved, partially resolved, needs escalation, or unresolved. If Unresolved, explain the reason briefly.")
    improvement_suggestions: str = Field(description="Provide at least one specific recommendation to improve customer experience. Avoid vague suggestions.")


AUDIO_ANALYSIS_PROMPT = """
You are an AI assistant analyzing call center audio recordings for Renault Korea. Listen to the audio and provide a comprehensive analysis.

""" + (f"""
ISSUE CATEGORIZATION RULES:
{ISSUE_CATEGORIZATION_PROMPT}

When categorizing key_issues:
- Use EXACT category names from the CATEGORIES list above
- Apply all EXCLUSION RULES and OVERRIDE RULES
- Multiple categories are allowed if directly applicable
- If no category applies after applying all rules, use: ["Confirmation needed"]
- Do NOT invent categories outside the predefined list

""" if ISSUE_CATEGORIZATION_PROMPT else "") + """
IMPORTANT - Prevent hallucination:
- If the audio is too short (less than 3 exchanges) or contains minimal information, state "Insufficient audio data" in summaries.
- If the content is not a customer service conversation, state "Not a customer service call" in summaries.
- If you cannot determine key details with confidence, state "Unclear from audio" rather than guessing.
- Only include information explicitly stated or strongly implied in the audio.
- NEVER invent customer issues, sentiments, or resolutions that aren't evidenced in the audio.

Ensure that:
- Every statement in your analysis must be directly supported by content in the audio.
- Customer sentiment analysis tracks changes throughout the call rather than assigning a fixed sentiment.
- Agent understanding assessment includes specific examples that demonstrate their comprehension of vehicle concerns.
- Improvement suggestions are practical and actionable, with a focus on addressing any gaps in advisor understanding.
"""


class GeminiAudioAnalyzer:
    """
    Handles audio analysis using Gemini 2.5 Flash Lite with direct audio input
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash-lite",
        max_workers: int = 16,
        max_retries: int = 3
    ):
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.max_workers = max_workers
        self.max_retries = max_retries
        
        logger.info(f"Initialized GeminiAudioAnalyzer with model: {model}, workers: {max_workers}")
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze a single audio file with Gemini using Files API
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with transcription and analysis
        """
        uploaded_file = None
        try:
            # Upload file to Gemini Files API
            logger.debug(f"Uploading {audio_path} to Files API...")
            uploaded_file = self.client.files.upload(file=audio_path)
            
            # Call Gemini API with structured output using Pydantic schema
            response = self.client.models.generate_content(
                model=self.model,
                contents=[AUDIO_ANALYSIS_PROMPT, uploaded_file],
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": CallAnalysis.model_json_schema(),
                }
            )
            
            # Parse response using Pydantic model for validation
            analysis = CallAnalysis.model_validate_json(response.text)
            
            # Convert to dict and return as JSON string
            return json.dumps(analysis.model_dump(), ensure_ascii=False)
            
        except Exception as e:
            logger.error(f"Analysis failed for {audio_path}: {e}")
            raise
        finally:
            # Clean up uploaded file
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.debug(f"Deleted uploaded file: {uploaded_file.name}")
                except Exception as e:
                    logger.warning(f"Failed to delete uploaded file: {e}")
    
    def analyze_batch(
        self,
        input_dir: str,
        output_transcriptions: str = "transcriptions.json",
        output_analysis: str = "analysis.json",
        checkpoint_file: str = "gemini_progress.json",
        save_interval: int = 10
    ) -> tuple[Dict[str, str], Dict[str, str]]:
        """
        Analyze all audio files in a directory in parallel
        
        Args:
            input_dir: Directory containing audio files
            output_transcriptions: Path to save transcriptions
            output_analysis: Path to save analysis
            checkpoint_file: Path to save progress
            save_interval: Save checkpoint every N files
            
        Returns:
            Tuple of (transcriptions dict, analysis dict)
        """
        import json
        
        # Load existing results
        transcriptions = {}
        analyses = {}
        
        if os.path.exists(output_transcriptions):
            transcriptions = load_json(output_transcriptions)
            logger.info(f"Loaded {len(transcriptions)} existing transcriptions (for checkpoint resume)")
        
        if os.path.exists(output_analysis):
            analyses = load_json(output_analysis)
            logger.info(f"Loaded {len(analyses)} existing analyses (for checkpoint resume)")
        
        tracker = ProgressTracker(checkpoint_file)
        
        # Get all audio files
        audio_files = get_audio_files(input_dir)
        
        if not audio_files:
            logger.warning(f"No audio files found in {input_dir}")
            return transcriptions, analyses
        
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
            return transcriptions, analyses
        
        # Process files in parallel
        processing_times = []
        files_processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_with_retry, audio_path): audio_path
                for audio_path in remaining_files
            }
            
            with tqdm(total=len(remaining_files), desc="Analyzing", unit="file") as pbar:
                for future in as_completed(future_to_file):
                    # Check for shutdown request
                    if shutdown_requested:
                        logger.info("Cancelling remaining tasks...")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    
                    audio_path = future_to_file[future]
                    filename = os.path.basename(audio_path)
                    
                    try:
                        result = future.result()
                        
                        if result:
                            # Parse JSON result
                            try:
                                data = json.loads(result)
                                
                                # Extract transcription and analysis
                                if 'transcription' in data:
                                    transcriptions[filename] = data['transcription']
                                    del data['transcription']  # Remove from analysis dict
                                else:
                                    transcriptions[filename] = "No transcription available"
                                
                                analyses[filename] = json.dumps(data, ensure_ascii=False)
                                tracker.mark_processed(filename)
                            
                            except json.JSONDecodeError as e:
                                logger.error(f"JSON parsing error for {filename}: {e}")
                                logger.debug(f"Raw response sample: {result[:500]}")
                                
                                # Try to salvage what we can - save as raw text
                                transcriptions[filename] = result[:2000] if len(result) > 2000 else result
                                analyses[filename] = json.dumps({
                                    "error": "JSON parsing failed",
                                    "raw_response_sample": result[:500]
                                }, ensure_ascii=False)
                                tracker.mark_processed(filename)  # Mark as processed but with error
                        
                        else:
                            tracker.mark_failed(filename)
                            transcriptions[filename] = "Analysis failed"
                            analyses[filename] = json.dumps({"error": "Analysis failed"})
                    
                    except Exception as e:
                        logger.error(f"Error processing {filename}: {e}")
                        tracker.mark_failed(filename)
                        transcriptions[filename] = "Processing error"
                        analyses[filename] = json.dumps({"error": str(e)})
                    
                    files_processed += 1
                    pbar.update(1)
                    
                    # Save checkpoint
                    if files_processed % save_interval == 0:
                        save_json(transcriptions, output_transcriptions)
                        save_json(analyses, output_analysis)
                        tracker.save_progress()
                        logger.info(f"Checkpoint saved after {files_processed} files")
        
        # Final save
        save_json(transcriptions, output_transcriptions)
        save_json(analyses, output_analysis)
        tracker.save_progress()
        
        final_stats = tracker.get_stats()
        
        if shutdown_requested:
            logger.info(f"Shutdown completed. Processed {files_processed}/{len(remaining_files)} files before interruption.")
        else:
            logger.info(f"Analysis complete!")
        
        logger.info(f"Total processed: {final_stats['processed']}")
        logger.info(f"Total failed: {final_stats['failed']}")
        
        return transcriptions, analyses
    
    def _process_with_retry(self, audio_path: str) -> Optional[str]:
        """Process with retry logic"""
        for attempt in range(self.max_retries):
            try:
                return self.analyze_audio(audio_path)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} for {os.path.basename(audio_path)}: {e}")
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
                else:
                    logger.error(f"Failed after {self.max_retries} attempts: {audio_path}")
                    return None
