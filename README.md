# Call Center Analytics Platform

A production-ready AI-powered call center analytics system for Renault Korea that analyzes audio recordings and generates actionable insights using Google's Gemini 2.5 Flash Lite with structured output.

## Features

- 🎙️ **Direct Audio Analysis**: Gemini 2.5 Flash Lite analyzes audio directly using Files API - no separate transcription needed
- 🤖 **Structured Output**: Type-safe analysis with Pydantic models ensuring consistent JSON schema
- 📊 **Visualizations**: Comprehensive charts including top issue categories, sentiment trajectories, and resolution status
- 📝 **Reporting**: Export results to Excel, CSV, and Word documents with proper Korean text support
- 🚀 **Parallel Processing**: 32 concurrent workers for fast batch processing (200 files in ~5-10 minutes)
- 🔄 **Checkpoint Recovery**: Automatic progress tracking with resume capability
- 💾 **Issue Categorization**: ~200 predefined vehicle issue categories with exclusion/override rules
- ⚙️ **CLI Interface**: Simple command-line interface for all operations

## Project Structure

```
call-center-analytics/
├── src/
│   ├── __init__.py
│   ├── gemini_audio.py        # Gemini direct audio analysis with Files API
│   ├── analysis.py            # Legacy Gemini text analysis (deprecated)
│   ├── transcription.py       # Legacy Whisper transcription (deprecated)
│   ├── database.py            # Database operations
│   ├── reporting.py           # Report generation (Excel, Word, CSV, JSON)
│   ├── visualization.py       # Chart and graph generation
│   └── utils.py               # Utility functions
├── config/
│   ├── config.yaml            # Configuration settings
│   └── prompt.py              # Vehicle issue categorization rules (~200 categories)
├── tests/
│   ├── __init__.py
│   ├── test_transcription.py
│   └── test_analysis.py
├── data/
│   ├── audio/                 # Input audio files (WAV format, organized by duration)
│   ├── output/                # Generated reports and visualizations
│   └── database/              # SQLite database
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore rules
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
└── README.md                  # This file
```

## Installation

### Option 1: Local Installation

1. **Clone or navigate to the project directory**
   ```bash
   cd call-center-analytics
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   copy .env.example .env  # Windows
   # cp .env.example .env  # Linux/Mac
   ```
   
   Edit `.env` and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Option 2: Docker Installation

```bash
docker-compose up -d
```

## Usage

### Recommended: Gemini Pipeline (Fast & Direct)

Process all audio files with Gemini 2.5 Flash Lite (direct audio analysis, no transcription step):

```bash
python main.py gemini-pipeline --input-dir audio --output-dir data/output --workers 32
```

This will:
1. Analyze all audio files in parallel (32 workers)
2. Extract transcription + analysis in a single API call
3. Save results to `data/output/transcriptions.json` and `data/output/analysis.json`
4. Generate Excel/Word reports
5. Create visualizations

### Generate Reports Only

```bash
python main.py report --transcriptions data/output/transcriptions.json --analysis data/output/analysis.json --audio-dir audio --output data/output/call_center_analysis --format excel
```

Supported formats: `excel`, `csv`, `word`, `json`, `all`

### Generate Visualizations Only

```bash
python main.py visualize --transcriptions data/output/transcriptions.json --analysis data/output/analysis.json --audio-dir audio --output-dir data/output/visualizations
```

### Legacy Pipeline (Deprecated)

The old Whisper → Gemini pipeline is still available but slower:

```bash
# 1. Transcribe with Whisper
python main.py transcribe --input-dir audio --output data/output/transcriptions.json

# 2. Analyze transcriptions
python main.py analyze --transcriptions data/output/transcriptions.json --output data/output/analysis.json

# 3. Full pipeline
python main.py pipeline --input-dir audio --output-dir data/output
```

## Configuration

Edit `config/config.yaml` to customize:

- Audio processing settings
- Gemini model parameters
- Visualization preferences
- Database configuration
- Logging levels

Example:
```yaml
gemini:
  model: "gemini-2.5-flash-lite"
  max_workers: 32
  retry_attempts: 3

transcription:
  model: "openai/whisper-large-v3-turbo"  # Legacy - not used by gemini-pipeline
  language: "Korean"
  batch_size: 24

database:
  path: "data/database/analytics.db"

visualization:
  style: "ggplot"
  dpi: 150
```

### Issue Categorization

Vehicle issue categories are defined in `config/prompt.py` with ~200 specific categories including:
- Navigation and infotainment issues
- Mechanical problems (engine, transmission, CV joints)
- Electrical issues (battery, sensors, warning lights)
- ADAS and safety features
- Comfort and convenience features

The system uses exclusion and override rules to accurately categorize customer concerns.

## API Keys Setup

### Gemini API Key (Required)
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create an API key
3. Add to `.env` file as `GEMINI_API_KEY=your_key_here`

**Note**: HuggingFace token is no longer required for the Gemini pipeline.

## Features Detail

### Direct Audio Analysis with Gemini
- **Files API Integration**: Uploads audio to Gemini, analyzes, then deletes
- **Structured Output**: Pydantic models ensure consistent JSON schema
- **No Transcription Errors**: Bypasses separate transcription step that could introduce errors
- **Parallel Processing**: 32 concurrent workers for fast batch processing
- **Checkpoint Recovery**: Automatic progress tracking with `gemini_progress.json`
- **Graceful Shutdown**: Ctrl+C saves progress before exiting

### Analysis Features
- **Customer Sentiment Trajectory**: 
  - "consistently neutral"
  - "consistently positive"  
  - "consistently negative"
  - "negative to positive"
- **Issue Classification**: ~200 vehicle-specific categories with override rules
- **Agent Performance Evaluation**: Professionalism, understanding, effectiveness
- **Resolution Status**: Resolved, Partially Resolved, Escalate to next step, Unresolved
- **Improvement Suggestions**: Specific, actionable recommendations

### Reporting
- **Excel**: Multi-column spreadsheet with proper Korean text (Unicode NFC normalization)
- **Word**: Formatted documents with Malgun Gothic font for Korean
- **CSV**: Compatible with external tools
- **JSON**: Programmatic access to raw data
- **Pipe-Separated Categories**: Prevents splitting categories that contain commas

### Visualizations
- **Call Duration Distribution**: Histogram with symlog scale for long calls
- **Sentiment Trajectory**: Pie chart of 4 sentiment categories with custom colors
- **Resolution Status**: Bar chart (excludes N/A values)
- **Top 15 Issue Categories**: Horizontal bar chart with full category names
- **Issue Categories Pie Chart**: Top 12 categories with "Other" for remainder
- **Word Cloud**: Common issues visualization
- **Duration vs Resolution**: Scatter plot with sentiment coloring

All charts properly handle categories with commas (e.g., "Instrument cluster warning light on (STOP, wrench)").

## Troubleshooting

### JSON Parsing Errors (Resolved)
The system now uses **structured output** with Pydantic models, eliminating JSON parsing errors that occurred with the old text-based approach.

### Category Name Truncation (Resolved)
Categories are now separated by ` | ` instead of `,` to prevent splitting names that contain commas internally.

### API Rate Limits
Adjust concurrency:
```bash
python main.py gemini-pipeline --input-dir audio --output-dir data/output --workers 16
```

Or in `config/config.yaml`:
```yaml
gemini:
  max_workers: 16  # Reduce from 32
```

### Checkpoint Recovery
If processing is interrupted, simply run the same command again:
```bash
python main.py gemini-pipeline --input-dir audio --output-dir data/output --workers 32
```

The system will:
1. Load existing results from `gemini_progress.json`
2. Skip already-processed files
3. Resume from where it left off

To start fresh, delete `gemini_progress.json`.

## Development

### Running Tests

```bash
pytest tests/
```

### Code Formatting

```bash
black src/ tests/
flake8 src/ tests/
```

## Technology Stack

- **Google Gemini 2.5 Flash Lite**: Direct audio analysis with structured output
- **Pydantic**: Type-safe schema validation
- **Files API**: Reliable audio upload/analysis/cleanup
- **Pandas**: Data manipulation and reporting
- **Matplotlib**: Visualization generation
- **OpenPyXL**: Excel file generation
- **Python-docx**: Word document generation

## Performance

- **200 audio files**: ~5-10 minutes with 32 workers
- **Parallel processing**: ThreadPoolExecutor with 32 concurrent API calls
- **Memory efficient**: Files uploaded to Gemini, not loaded in memory
- **Checkpoint recovery**: Resume from interruptions without re-processing

## License

MIT License

## Support

For issues and questions, please open an issue on [GitHub](https://github.com/rodolphethinks/call-center-analysis).

## Acknowledgments

- Google Gemini AI for advanced audio analysis
- Pydantic for structured output validation
- The open-source Python community
