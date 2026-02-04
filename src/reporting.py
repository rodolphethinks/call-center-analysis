"""
Report generation module for Excel, CSV, and Word documents
"""

import logging
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from docx import Document
from docx.shared import Pt
from datetime import datetime

from .utils import ensure_directory, parse_call_id_datetime


logger = logging.getLogger(__name__)


class ReportGenerator:
    """
    Generates reports in various formats (Excel, CSV, Word, JSON)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.report_config = self.config.get('reporting', {})
    
    def _parse_analysis_json(self, analysis_text: str) -> Dict[str, Any]:
        """Parse analysis JSON from text"""
        try:
            # Try to extract JSON from markdown code blocks
            if '```json' in analysis_text:
                start = analysis_text.find('```json') + 7
                end = analysis_text.find('```', start)
                json_str = analysis_text[start:end].strip()
            elif '```' in analysis_text:
                start = analysis_text.find('```') + 3
                end = analysis_text.find('```', start)
                json_str = analysis_text[start:end].strip()
            else:
                json_str = analysis_text
            
            return json.loads(json_str)
        except Exception as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")
            return {}
    
    def create_dataframe(
        self,
        transcriptions: Dict[str, str],
        analyses: Dict[str, str],
        audio_dir: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Create a pandas DataFrame from transcriptions and analyses
        
        Args:
            transcriptions: Dictionary of call_id -> transcription
            analyses: Dictionary of call_id -> analysis JSON
            audio_dir: Optional directory path to get audio durations
            
        Returns:
            DataFrame with all analysis data
        """
        data_list = []
        
        for call_id, analysis_text in analyses.items():
            try:
                # Parse analysis JSON
                analysis = self._parse_analysis_json(analysis_text)
                
                # Get transcription
                transcription = transcriptions.get(call_id, "N/A")
                
                # Parse call datetime from ID
                call_datetime = parse_call_id_datetime(call_id)
                
                # Get audio duration if audio_dir provided
                duration = None
                if audio_dir:
                    try:
                        import soundfile as sf
                        import os
                        from pathlib import Path
                        
                        # Search for audio file recursively in subdirectories
                        audio_path = None
                        for root, dirs, files in os.walk(audio_dir):
                            if call_id in files:
                                audio_path = os.path.join(root, call_id)
                                break
                        
                        if audio_path and os.path.exists(audio_path):
                            with sf.SoundFile(audio_path) as f:
                                duration = len(f) / f.samplerate
                    except Exception as e:
                        logger.debug(f"Could not get duration for {call_id}: {e}")
                
                # Helper to normalize Unicode strings for Korean text
                def normalize_text(text):
                    if text is None or text == "N/A":
                        return "N/A"
                    import unicodedata
                    # Ensure proper Unicode normalization for Korean characters
                    normalized = unicodedata.normalize('NFC', str(text))
                    return normalized
                
                # Build row
                row = {
                    "Call ID": call_id,
                    "Date": call_datetime,
                    "Duration": duration,
                    "Customer Summary": normalize_text(analysis.get("customer_summary", "N/A")),
                    "Agent Summary": normalize_text(analysis.get("agent_summary", "N/A")),
                    "Customer Sentiment Trajectory": normalize_text(analysis.get("customer_sentiment_trajectory", "N/A")),
                    "Customer Sentiment Examples": normalize_text(analysis.get("customer_sentiment_examples", "N/A")),
                    "Key Issues": normalize_text(" | ".join(analysis.get("key_issues", [])) if isinstance(analysis.get("key_issues"), list) else analysis.get("key_issues", "N/A")),
                    "Agent Performance Overall": normalize_text(analysis.get("agent_performance_overall", "N/A")),
                    "Agent Understanding": normalize_text(analysis.get("agent_understanding", "N/A")),
                    "Resolution Status": normalize_text(analysis.get("resolution_status", "N/A")),
                    "Improvement Suggestions": normalize_text(analysis.get("improvement_suggestions", "N/A")),
                    "Transcription": normalize_text(transcription)
                }
                
            except Exception as e:
                logger.error(f"Error processing {call_id}: {e}")
                row = {
                    "Call ID": call_id,
                    "Date": parse_call_id_datetime(call_id),
                    "Duration": None,
                    "Customer Summary": f"Error: {e}",
                    "Agent Summary": "N/A",
                    "Customer Sentiment Trajectory": "N/A",
                    "Customer Sentiment Examples": "N/A",
                    "Key Issues": "N/A",
                    "Agent Performance Overall": "N/A",
                    "Agent Understanding": "N/A",
                    "Resolution Status": "N/A",
                    "Improvement Suggestions": "N/A",
                    "Transcription": transcriptions.get(call_id, "N/A")
                }
            
            data_list.append(row)
        
        df = pd.DataFrame(data_list)
        
        # Sort by date if available
        if 'Date' in df.columns:
            df = df.sort_values('Date', ascending=False)
        
        # Normalize sentiment categories
        if 'Customer Sentiment Trajectory' in df.columns:
            df['Customer Sentiment Trajectory'] = df['Customer Sentiment Trajectory'].replace({
                'consistently frustrated': 'consistently negative',
                'positive to negative': 'consistently negative'
            })
        
        return df
    
    def export_to_excel(
        self,
        df: pd.DataFrame,
        output_path: str,
        include_summary: bool = True
    ) -> None:
        """
        Export DataFrame to Excel with formatting
        
        Args:
            df: DataFrame to export
            output_path: Path to save Excel file
            include_summary: Whether to include summary sheet
        """
        ensure_directory(str(Path(output_path).parent))
        
        # Ensure all string columns are properly encoded as UTF-8
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str)
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Call Analysis', index=False)
            
            # Get the worksheet to set column widths
            worksheet = writer.sheets['Call Analysis']
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter
                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except:
                        pass
                adjusted_width = min(max_length + 2, 50)  # Cap at 50
                worksheet.column_dimensions[column_letter].width = adjusted_width
            
            # Summary sheet
            if include_summary:
                summary_data = []
                
                # Total calls
                summary_data.append({"Metric": "Total Calls", "Value": len(df)})
                
                # Sentiment distribution
                if 'Customer Sentiment Trajectory' in df.columns:
                    sentiment_counts = df['Customer Sentiment Trajectory'].value_counts()
                    for sentiment, count in sentiment_counts.items():
                        summary_data.append({
                            "Metric": f"Sentiment - {sentiment}",
                            "Value": count
                        })
                
                # Resolution distribution
                if 'Resolution Status' in df.columns:
                    resolution_counts = df['Resolution Status'].value_counts()
                    for status, count in resolution_counts.items():
                        summary_data.append({
                            "Metric": f"Resolution - {status}",
                            "Value": count
                        })
                
                # Average duration
                if 'Duration' in df.columns and df['Duration'].notna().any():
                    avg_duration = df['Duration'].mean()
                    summary_data.append({
                        "Metric": "Average Call Duration (seconds)",
                        "Value": f"{avg_duration:.2f}"
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Excel report saved to {output_path}")
    
    def export_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """Export DataFrame to CSV"""
        ensure_directory(str(Path(output_path).parent))
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"CSV report saved to {output_path}")
    
    def export_to_word(
        self,
        df: pd.DataFrame,
        output_path: str,
        max_calls: Optional[int] = None
    ) -> None:
        """
        Export DataFrame to Word document
        
        Args:
            df: DataFrame to export
            output_path: Path to save Word document
            max_calls: Maximum number of calls to include (None for all)
        """
        ensure_directory(str(Path(output_path).parent))
        
        # Create document with proper encoding support
        doc = Document()
        
        # Set default font to one that supports Korean characters
        style = doc.styles['Normal']
        font = style.font
        font.name = 'Malgun Gothic'  # Korean-supporting font
        font.size = Pt(10)
        
        doc.add_heading("Call Center Analysis Report", level=1)
        doc.add_paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        doc.add_paragraph(f"Total Calls: {len(df)}")
        doc.add_paragraph("")
        
        # Limit number of calls if specified
        df_subset = df.head(max_calls) if max_calls else df
        
        for idx, row in df_subset.iterrows():
            # Call header
            doc.add_heading(f"Call ID: {row.get('Call ID', 'N/A')}", level=2)
            
            # Helper function to add bold label with text
            def add_field(label: str, value: Any):
                p = doc.add_paragraph()
                p.add_run(f"{label}: ").bold = True
                # Ensure value is properly converted to string with UTF-8
                text_value = str(value) if value is not None else 'N/A'
                run = p.add_run(text_value)
                run.font.name = 'Malgun Gothic'  # Korean-supporting font
            
            # Add all fields
            if 'Date' in row and pd.notna(row['Date']):
                add_field("Date", row['Date'])
            
            if 'Duration' in row and pd.notna(row['Duration']):
                add_field("Duration", f"{row['Duration']:.2f} seconds")
            
            add_field("Customer Summary", row.get('Customer Summary', 'N/A'))
            add_field("Agent Summary", row.get('Agent Summary', 'N/A'))
            add_field("Sentiment Trajectory", row.get('Customer Sentiment Trajectory', 'N/A'))
            add_field("Sentiment Examples", row.get('Customer Sentiment Examples', 'N/A'))
            add_field("Key Issues", row.get('Key Issues', 'N/A'))
            add_field("Agent Performance", row.get('Agent Performance Overall', 'N/A'))
            add_field("Agent Understanding", row.get('Agent Understanding', 'N/A'))
            add_field("Resolution Status", row.get('Resolution Status', 'N/A'))
            add_field("Improvement Suggestions", row.get('Improvement Suggestions', 'N/A'))
            
            # Add separator
            doc.add_paragraph("\n" + "-" * 80 + "\n")
        
        doc.save(output_path)
        logger.info(f"Word report saved to {output_path}")
    
    def export_to_json(
        self,
        transcriptions: Dict[str, str],
        analyses: Dict[str, str],
        output_path: str
    ) -> None:
        """Export data to JSON format"""
        ensure_directory(str(Path(output_path).parent))
        
        data = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "total_calls": len(analyses)
            },
            "calls": []
        }
        
        for call_id, analysis_text in analyses.items():
            analysis = self._parse_analysis_json(analysis_text)
            
            call_data = {
                "call_id": call_id,
                "date": parse_call_id_datetime(call_id).isoformat() if parse_call_id_datetime(call_id) else None,
                "transcription": transcriptions.get(call_id),
                "analysis": analysis
            }
            
            data["calls"].append(call_data)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"JSON report saved to {output_path}")
    
    def generate_all_reports(
        self,
        transcriptions: Dict[str, str],
        analyses: Dict[str, str],
        output_dir: str,
        audio_dir: Optional[str] = None,
        base_filename: str = "call_center_analysis"
    ) -> Dict[str, str]:
        """
        Generate reports in all formats
        
        Args:
            transcriptions: Dictionary of transcriptions
            analyses: Dictionary of analyses
            output_dir: Directory to save reports
            audio_dir: Optional audio directory for durations
            base_filename: Base name for output files
            
        Returns:
            Dictionary mapping format to file path
        """
        ensure_directory(output_dir)
        
        # Create DataFrame
        df = self.create_dataframe(transcriptions, analyses, audio_dir)
        
        # Generate reports
        outputs = {}
        
        # Excel
        excel_path = str(Path(output_dir) / f"{base_filename}.xlsx")
        self.export_to_excel(df, excel_path)
        outputs['excel'] = excel_path
        
        # CSV
        csv_path = str(Path(output_dir) / f"{base_filename}.csv")
        self.export_to_csv(df, csv_path)
        outputs['csv'] = csv_path
        
        # Word (limit to first 50 calls)
        word_path = str(Path(output_dir) / f"{base_filename}.docx")
        self.export_to_word(df, word_path, max_calls=50)
        outputs['word'] = word_path
        
        # JSON
        json_path = str(Path(output_dir) / f"{base_filename}.json")
        self.export_to_json(transcriptions, analyses, json_path)
        outputs['json'] = json_path
        
        logger.info(f"All reports generated in {output_dir}")
        
        return outputs
