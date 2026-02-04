# Call Center Analytics Platform

A production-ready AI-powered call center analytics system that transcribes audio recordings, analyzes customer interactions, and generates actionable insights.

## Features

- 🎙️ **Audio Transcription**: Automatic speech-to-text using Whisper AI (supports Korean and other languages)
- 🤖 **AI Analysis**: Sentiment analysis, issue classification, and agent performance evaluation using Gemini AI
- 📊 **Visualizations**: Comprehensive charts and dashboards for insights
- 📝 **Reporting**: Export results to Excel, CSV, and Word documents
- 💾 **Database**: SQLite storage for historical data and analytics
- 🚀 **Async Processing**: Fast parallel processing for large datasets
- 🐳 **Docker Support**: Easy deployment with containerization
- ⚙️ **CLI Interface**: Simple command-line interface for all operations

## Project Structure

```
call-center-analytics/
├── src/
│   ├── __init__.py
│   ├── transcription.py      # Audio transcription logic
│   ├── analysis.py            # AI-powered conversation analysis
│   ├── database.py            # Database operations
│   ├── reporting.py           # Report generation (Excel, Word)
│   ├── visualization.py       # Chart and graph generation
│   └── utils.py               # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_transcription.py
│   └── test_analysis.py
├── data/
│   ├── audio/                 # Input audio files
│   ├── output/                # Generated reports
│   └── database/              # SQLite database
├── config/
│   └── config.yaml            # Configuration settings
├── main.py                    # CLI entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
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
   
   Edit `.env` and add your API keys:
   ```
   GEMINI_API_KEY=your_gemini_api_key_here
   HUGGINGFACE_TOKEN=your_huggingface_token_here
   ```

### Option 2: Docker Installation

```bash
docker-compose up -d
```

## Usage

### 1. Transcribe Audio Files

```bash
python main.py transcribe --input-dir data/audio --output transcriptions.json
```

### 2. Analyze Transcriptions

```bash
python main.py analyze --input transcriptions.json --output analysis.json
```

### 3. Generate Reports

```bash
python main.py report --input analysis.json --format excel --output data/output/report.xlsx
```

### 4. Generate Visualizations

```bash
python main.py visualize --input analysis.json --output-dir data/output/charts
```

### 5. Run Full Pipeline

```bash
python main.py pipeline --input-dir data/audio --output-dir data/output
```

## Configuration

Edit `config/config.yaml` to customize:

- Audio processing settings
- AI model parameters
- Visualization preferences
- Database configuration
- Logging levels

Example:
```yaml
transcription:
  model: "openai/whisper-large-v3-turbo"
  language: "Korean"
  batch_size: 24
  chunk_length: 30

analysis:
  model: "gemini-2.0-flash"
  max_workers: 32
  retry_attempts: 3

database:
  path: "data/database/analytics.db"
```

## API Keys Setup

### Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create an API key
3. Add to `.env` file

### HuggingFace Token
1. Visit [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Create a read token
3. Add to `.env` file

## Features Detail

### Transcription
- Automatic language detection
- Quality verification with AI
- Progress tracking and checkpointing
- Retry logic for failed files
- Support for WAV, MP3, and other audio formats

### Analysis
- Customer sentiment trajectory (positive/negative trends)
- Issue classification (Customer, Vehicle, After-Sales, Other)
- Agent performance evaluation
- Resolution status tracking
- Improvement suggestions

### Reporting
- Excel spreadsheets with multiple sheets
- Word documents with formatted analysis
- CSV exports for external tools
- JSON output for programmatic access

### Visualizations
- Call duration distributions
- Sentiment trajectory charts
- Resolution status breakdowns
- Issue frequency analysis
- Agent performance metrics
- Word clouds for common issues

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

## Troubleshooting

### CUDA/GPU Issues
If you don't have a GPU, the system will automatically use CPU. To force CPU:
```bash
python main.py transcribe --input-dir data/audio --device cpu
```

### Out of Memory
Reduce batch size in `config/config.yaml`:
```yaml
transcription:
  batch_size: 8  # Reduce from default 24
```

### API Rate Limits
Adjust concurrency in config:
```yaml
analysis:
  max_workers: 8  # Reduce from default 32
  delay_between_requests: 0.1  # Increase delay
```

## License

MIT License

## Support

For issues and questions, please open an issue on GitHub.

## Acknowledgments

- OpenAI Whisper for speech recognition
- Google Gemini for AI analysis
- HuggingFace for model hosting
