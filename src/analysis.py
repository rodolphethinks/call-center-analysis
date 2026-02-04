"""
Conversation analysis module using Gemini AI
"""

import logging
import time
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from tenacity import retry, wait_exponential, stop_after_attempt
from google import genai
import sys
from pathlib import Path

# Add config directory to path
config_dir = Path(__file__).parent.parent / 'config'
sys.path.insert(0, str(config_dir))

try:
    from prompt import KEYWORDS as ISSUE_CATEGORIZATION_PROMPT
except ImportError:
    logger.warning("Could not import issue categorization prompt from config/prompt.py")
    ISSUE_CATEGORIZATION_PROMPT = ""

from .utils import save_json, load_json


logger = logging.getLogger(__name__)


ANALYSIS_INSTRUCTION = """
You are an AI assistant analyzing call center conversations for Renault Korea. Your goal is to extract actionable insights from the transcription. 
Analyze the following conversation and provide a structured JSON output strictly following this format:

{
  "customer_summary": "Briefly summarize the customer's statements in 1-2 sentences.",
  "agent_summary": "Briefly summarize the agent's statements in 1-2 sentences.",
  "customer_sentiment_trajectory": "Classify the customer's emotional trajectory ONLY as one of these FOUR options: 'consistently neutral', 'consistently positive', 'consistently negative', or 'negative to positive'. Do NOT use any other sentiment labels.",
  "customer_sentiment_examples": "Provide specific examples from the conversation that demonstrate this emotional path, based on tone of voice, word choice, and reaction to resolution attempts.",
  "key_issues": ["List of categorized issues from the CATEGORIES section below. Use EXACT category names. Multiple categories allowed if applicable."],
  "agent_performance_overall": "Assess the agent's overall effectiveness in handling the call (e.g., professionalism, responsiveness, clarity).",
  "agent_understanding": "Specifically evaluate how well the advisor understood the customer's vehicle concerns. Provide concrete examples from the conversation that demonstrate the level of understanding. This will help identify training opportunities.",
  "resolution_status": "State if the issue was either: 'Resolved', 'Partially Resolved', 'Escalate to next step', or 'Unresolved'. If Unresolved, explain the reason briefly.",
  "improvement_suggestions": "Provide at least one **specific** recommendation to improve customer experience. Avoid vague suggestions."
}

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
- If the transcription is too short (less than 3 exchanges) or contains minimal information, respond with: {"analysis_status": "insufficient_data", "explanation": "The provided transcription is too brief to perform reliable analysis without risking fabrication."} Do NOT attempt to analyze insufficient data.
- If the content is not a customer service conversation, respond with: {"analysis_status": "irrelevant_content", "explanation": "The provided text does not appear to be a customer service conversation related to Renault Korea."} 
- If you cannot determine key details with confidence, state "Unclear from transcript" rather than guessing.
- Only include information explicitly stated or strongly implied in the transcript.
- NEVER invent customer issues, sentiments, or resolutions that aren't evidenced in the transcript.

Ensure that:
- The JSON output is **well-structured** and follows the format exactly.
- Every statement in your analysis must be directly supported by content in the transcript.
- Customer sentiment analysis tracks changes throughout the call rather than assigning a fixed sentiment. ONLY classify it using one of the FOUR given categories. Do not mention neutral as an output.
- Agent understanding assessment includes specific examples that demonstrate their comprehension of vehicle concerns.
- Improvement suggestions are practical and actionable, with a focus on addressing any gaps in advisor understanding.
"""


class ConversationAnalyzer:
    """
    Analyzes call center conversations using Gemini AI
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        max_workers: int = 32,
        delay_between_requests: float = 0.03,
        retry_attempts: int = 5,
        retry_min_wait: int = 1,
        retry_max_wait: int = 10
    ):
        self.model_name = model
        self.max_workers = max_workers
        self.delay_between_requests = delay_between_requests
        self.retry_attempts = retry_attempts
        self.retry_min_wait = retry_min_wait
        self.retry_max_wait = retry_max_wait
        
        # Configure Gemini
        self.client = genai.Client(api_key=api_key)
        self.model_name = model if 'gemini' in model else 'gemini-2.0-flash-exp'
        
        logger.info(f"Initialized ConversationAnalyzer with model: {model}")
    
    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(5)
    )
    def analyze_conversation(self, transcription: str) -> str:
        """
        Analyze a single conversation transcription
        
        Args:
            transcription: Text of the conversation
            
        Returns:
            JSON string with analysis results
        """
        try:
            full_prompt = f"{ANALYSIS_INSTRUCTION}\n\nNow, analyze the following transcription: {transcription}"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt
            )
            result = response.text.strip()
            
            # Clean up markdown code blocks if present
            if result.startswith('```json'):
                result = result[7:]  # Remove ```json
            if result.startswith('```'):
                result = result[3:]  # Remove ```
            if result.endswith('```'):
                result = result[:-3]  # Remove trailing ```
            
            return result.strip()
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise
    
    def analyze_batch(
        self,
        transcriptions: Dict[str, str],
        output_file: str = "analysis.json"
    ) -> Dict[str, str]:
        """
        Analyze multiple transcriptions in parallel
        
        Args:
            transcriptions: Dictionary mapping call IDs to transcriptions
            output_file: Path to save analysis results
            
        Returns:
            Dictionary mapping call IDs to analysis results
        """
        results = {}
        
        # Load existing results if available
        try:
            results = load_json(output_file)
            logger.info(f"Loaded {len(results)} existing analyses")
        except FileNotFoundError:
            pass
        
        # Filter out already analyzed transcriptions
        remaining = {
            call_id: text
            for call_id, text in transcriptions.items()
            if call_id not in results
        }
        
        if not remaining:
            logger.info("All transcriptions have been analyzed!")
            return results
        
        logger.info(f"Analyzing {len(remaining)} conversations...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {}
            
            for call_id, transcription in remaining.items():
                future = executor.submit(self.analyze_conversation, transcription)
                futures[future] = call_id
                time.sleep(self.delay_between_requests)
            
            with tqdm(total=len(futures), desc="Analyzing", unit="call") as pbar:
                for future in as_completed(futures):
                    call_id = futures[future]
                    
                    try:
                        analysis = future.result()
                        results[call_id] = analysis
                        
                    except Exception as e:
                        logger.error(f"Failed to analyze {call_id}: {e}")
                        results[call_id] = f"Analysis failed: {e}"
                    
                    pbar.update(1)
        
        # Save results
        save_json(results, output_file)
        logger.info(f"Analysis complete! Results saved to {output_file}")
        
        return results
    
    def classify_issues(
        self,
        issues: List[str],
        categories: List[str] = None
    ) -> Dict[str, List[str]]:
        """
        Classify issues into categories using AI
        
        Args:
            issues: List of issue descriptions
            categories: List of category names
            
        Returns:
            Dictionary mapping categories to lists of issues
        """
        if categories is None:
            categories = ["Customer Issues", "Vehicle Issues", "AS Issues", "Other Issues"]
        
        instruction = (
            f"You are a service analyst. Classify each issue into one of the following categories:\n"
            f"{', '.join(categories)}\n\n"
            f"Return your response in this format ONLY:\n"
            f"Category: Issue Description\n"
            f"For example: 'Vehicle Issues: Engine knocking sound'.\n"
            f"Keep it short and professional."
        )
        
        classified = {cat: [] for cat in categories}
        
        for issue in tqdm(issues, desc="Classifying issues"):
            try:
                prompt = f"{instruction}\n\nClassify this issue: \"{issue}\""
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt
                )
                result = response.text.strip()
                
                if ':' in result:
                    category, cleaned_issue = [x.strip() for x in result.split(':', 1)]
                    if category in classified:
                        classified[category].append(cleaned_issue)
                    else:
                        classified['Other Issues'].append(issue)
                else:
                    classified['Other Issues'].append(issue)
                    
            except Exception as e:
                logger.warning(f"Failed to classify issue '{issue}': {e}")
                classified['Other Issues'].append(issue)
            
            time.sleep(self.delay_between_requests)
        
        return classified
    
    def summarize_improvement(self, suggestion: str) -> str:
        """
        Summarize a detailed improvement suggestion into a short phrase
        
        Args:
            suggestion: Detailed improvement suggestion
            
        Returns:
            Short summary (2-5 words)
        """
        instruction = (
            "You are an analyst summarizing customer service improvement suggestions. "
            "Your task is to extract a **general and short** improvement suggestion (2 to 5 words) "
            "from each detailed description. Keep the improvement actionable and general enough to "
            "group with others (e.g., 'Faster response time', 'Better parts info', "
            "'Improved warranty tracking'). Avoid specific models, dates, or individual names. "
            "Only return the short improvement."
        )
        
        try:
            prompt = f"{instruction}\n\nDetailed suggestion: {suggestion}\n\nShort improvement:"
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text.strip()
            
        except Exception as e:
            logger.warning(f"Failed to summarize suggestion: {e}")
            return "General improvement needed"


def create_analyzer(config: Dict, env_vars: Dict) -> ConversationAnalyzer:
    """
    Factory function to create ConversationAnalyzer from config
    
    Args:
        config: Configuration dictionary
        env_vars: Environment variables dictionary
        
    Returns:
        ConversationAnalyzer instance
    """
    analysis_config = config.get('analysis', {})
    
    return ConversationAnalyzer(
        api_key=env_vars['GEMINI_API_KEY'],
        model=analysis_config.get('model', 'gemini-2.0-flash'),
        max_workers=analysis_config.get('max_workers', 32),
        delay_between_requests=analysis_config.get('delay_between_requests', 0.03),
        retry_attempts=analysis_config.get('retry_attempts', 5),
        retry_min_wait=analysis_config.get('retry_min_wait', 1),
        retry_max_wait=analysis_config.get('retry_max_wait', 10)
    )
