# Quick Start Guide

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   copy .env.example .env
   ```
   Edit `.env` and add your API keys.

3. **Place audio files** in `data/audio/` directory

## Run Full Pipeline

Process all audio files and generate complete analysis:

```bash
python main.py pipeline --input-dir data/audio --output-dir data/output
```

This will:
- Transcribe all audio files
- Analyze conversations with AI
- Generate Excel/CSV/Word/JSON reports
- Create visualizations and charts

## Individual Commands

### Transcribe Only
```bash
python main.py transcribe --input-dir data/audio --output transcriptions.json
```

### Analyze Only
```bash
python main.py analyze --input transcriptions.json --output analysis.json
```

### Generate Reports
```bash
python main.py report \
  --transcriptions transcriptions.json \
  --analysis analysis.json \
  --output report.xlsx \
  --format excel
```

### Generate Visualizations
```bash
python main.py visualize \
  --transcriptions transcriptions.json \
  --analysis analysis.json \
  --output-dir charts/
```

## Using Docker

```bash
# Build and run
docker-compose up

# Or run interactively
docker-compose run call-center-analytics python main.py --help
```

## Output Files

After running the pipeline, check `data/output/`:
- `transcriptions.json` - Audio transcriptions
- `analysis.json` - AI analysis results
- `report.xlsx` - Excel report with all data
- `report.csv` - CSV export
- `report.docx` - Word document report
- `report.json` - JSON export
- `charts/` - All visualizations (PNG files)

## Troubleshooting

### Missing API Keys
Make sure `.env` file contains:
```
GEMINI_API_KEY=your_key_here
HUGGINGFACE_TOKEN=your_token_here
```

### GPU Issues
To force CPU usage:
Edit `config/config.yaml`:
```yaml
transcription:
  device: "cpu"
```

### Memory Issues
Reduce batch size in `config/config.yaml`:
```yaml
transcription:
  batch_size: 8
```
