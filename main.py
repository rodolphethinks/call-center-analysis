"""
Main CLI entry point for Call Center Analytics Platform
"""

import click
import logging
from pathlib import Path

from src import (
    setup_logging,
    load_config,
    load_environment,
    load_json,
    create_transcriber,
    create_analyzer,
    create_database
)
from src.reporting import ReportGenerator
from src.visualization import DataVisualizer
from src.gemini_audio import GeminiAudioAnalyzer


@click.group()
@click.option('--config', default='config/config.yaml', help='Path to configuration file')
@click.option('--log-level', default='INFO', help='Logging level')
@click.pass_context
def cli(ctx, config, log_level):
    """Call Center Analytics Platform - AI-powered conversation analysis"""
    ctx.ensure_object(dict)
    
    # Setup logging
    setup_logging(log_level)
    
    # Load configuration and environment
    ctx.obj['config'] = load_config(config)
    ctx.obj['env'] = load_environment()
    ctx.obj['logger'] = logging.getLogger(__name__)
    
    ctx.obj['logger'].info("Call Center Analytics Platform initialized")


@cli.command()
@click.option('--input-dir', required=True, help='Directory containing audio files')
@click.option('--output', default='data/output/transcriptions.json', help='Output JSON file')
@click.option('--checkpoint', default='transcription_progress.json', help='Checkpoint file')
@click.option('--save-interval', default=10, help='Save checkpoint every N files')
@click.pass_context
def transcribe(ctx, input_dir, output, checkpoint, save_interval):
    """Transcribe audio files to text"""
    logger = ctx.obj['logger']
    logger.info(f"Starting transcription: {input_dir}")
    
    # Create transcriber
    transcriber = create_transcriber(ctx.obj['config'], ctx.obj['env'])
    
    # Transcribe files
    transcriptions = transcriber.transcribe_files(
        input_dir=input_dir,
        output_file=output,
        checkpoint_file=checkpoint,
        save_interval=save_interval
    )
    
    logger.info(f"Transcription complete! {len(transcriptions)} files processed")
    logger.info(f"Results saved to: {output}")


@cli.command()
@click.option('--input', required=True, help='Input transcriptions JSON file')
@click.option('--output', default='data/output/analysis.json', help='Output analysis JSON file')
@click.pass_context
def analyze(ctx, input, output):
    """Analyze transcriptions using AI"""
    logger = ctx.obj['logger']
    logger.info(f"Starting analysis: {input}")
    
    # Load transcriptions
    transcriptions = load_json(input)
    logger.info(f"Loaded {len(transcriptions)} transcriptions")
    
    # Create analyzer
    analyzer = create_analyzer(ctx.obj['config'], ctx.obj['env'])
    
    # Analyze
    analyses = analyzer.analyze_batch(transcriptions, output)
    
    logger.info(f"Analysis complete! {len(analyses)} calls analyzed")
    logger.info(f"Results saved to: {output}")


@cli.command()
@click.option('--transcriptions', required=True, help='Transcriptions JSON file')
@click.option('--analysis', required=True, help='Analysis JSON file')
@click.option('--output', required=True, help='Output file path')
@click.option('--format', type=click.Choice(['excel', 'csv', 'word', 'json', 'all']), 
              default='excel', help='Output format')
@click.option('--audio-dir', help='Audio directory for duration extraction')
@click.pass_context
def report(ctx, transcriptions, analysis, output, format, audio_dir):
    """Generate reports from analysis results"""
    logger = ctx.obj['logger']
    logger.info(f"Generating {format} report")
    
    # Load data
    transcriptions_data = load_json(transcriptions)
    analysis_data = load_json(analysis)
    
    # Create report generator
    generator = ReportGenerator(ctx.obj['config'])
    
    # Create DataFrame
    df = generator.create_dataframe(transcriptions_data, analysis_data, audio_dir)
    
    # Generate report(s)
    if format == 'all':
        output_dir = str(Path(output).parent)
        base_name = Path(output).stem
        outputs = generator.generate_all_reports(
            transcriptions_data,
            analysis_data,
            output_dir,
            audio_dir,
            base_name
        )
        for fmt, path in outputs.items():
            logger.info(f"{fmt.upper()}: {path}")
    elif format == 'excel':
        generator.export_to_excel(df, output)
        logger.info(f"Excel report saved: {output}")
    elif format == 'csv':
        generator.export_to_csv(df, output)
        logger.info(f"CSV report saved: {output}")
    elif format == 'word':
        generator.export_to_word(df, output)
        logger.info(f"Word report saved: {output}")
    elif format == 'json':
        generator.export_to_json(transcriptions_data, analysis_data, output)
        logger.info(f"JSON report saved: {output}")


@cli.command()
@click.option('--transcriptions', required=True, help='Transcriptions JSON file')
@click.option('--analysis', required=True, help='Analysis JSON file')
@click.option('--output-dir', default='data/output/charts', help='Output directory for charts')
@click.option('--audio-dir', help='Audio directory for duration extraction')
@click.pass_context
def visualize(ctx, transcriptions, analysis, output_dir, audio_dir):
    """Generate visualizations from analysis results"""
    logger = ctx.obj['logger']
    logger.info(f"Generating visualizations")
    
    # Load data
    transcriptions_data = load_json(transcriptions)
    analysis_data = load_json(analysis)
    
    # Create DataFrame
    generator = ReportGenerator(ctx.obj['config'])
    df = generator.create_dataframe(transcriptions_data, analysis_data, audio_dir)
    
    # Create visualizer
    visualizer = DataVisualizer(ctx.obj['config'])
    
    # Generate all visualizations
    outputs = visualizer.generate_all_visualizations(df, output_dir)
    
    logger.info(f"Generated {len(outputs)} visualizations in {output_dir}")
    for name, path in outputs.items():
        logger.info(f"  - {name}: {Path(path).name}")


@cli.command()
@click.option('--input-dir', required=True, help='Directory containing audio files')
@click.option('--output-dir', default='data/output', help='Output directory for results')
@click.option('--workers', default=16, help='Number of parallel workers')
@click.pass_context
def gemini_audio(ctx, input_dir, output_dir, workers):
    """Analyze audio files directly with Gemini (transcription + analysis in one step)"""
    logger = ctx.obj['logger']
    logger.info(f"Starting Gemini audio analysis: {input_dir}")
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create analyzer
    api_key = ctx.obj['env'].get('GEMINI_API_KEY')
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment")
        return
    
    analyzer = GeminiAudioAnalyzer(
        api_key=api_key,
        model="gemini-2.5-flash-lite",
        max_workers=workers
    )
    
    # Process all files
    transcriptions_file = str(Path(output_dir) / "transcriptions.json")
    analysis_file = str(Path(output_dir) / "analysis.json")
    
    transcriptions, analyses = analyzer.analyze_batch(
        input_dir=input_dir,
        output_transcriptions=transcriptions_file,
        output_analysis=analysis_file
    )
    
    logger.info(f"Gemini audio analysis complete!")
    logger.info(f"Processed {len(transcriptions)} files")
    logger.info(f"Transcriptions: {transcriptions_file}")
    logger.info(f"Analysis: {analysis_file}")


@cli.command()
@click.option('--input-dir', required=True, help='Directory containing audio files')
@click.option('--output-dir', default='data/output', help='Output directory for all results')
@click.option('--audio-dir', help='Audio directory (if different from input-dir)')
@click.option('--workers', default=16, help='Number of parallel workers for Gemini')
@click.pass_context
def gemini_pipeline(ctx, input_dir, output_dir, audio_dir, workers):
    """Run complete pipeline using Gemini audio analysis (faster, parallelized)"""
    logger = ctx.obj['logger']
    logger.info("Starting Gemini pipeline")
    
    if audio_dir is None:
        audio_dir = input_dir
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # File paths
    transcriptions_file = str(Path(output_dir) / "transcriptions.json")
    analysis_file = str(Path(output_dir) / "analysis.json")
    charts_dir = str(Path(output_dir) / "charts")
    
    # Step 1: Gemini Audio Analysis (transcription + analysis in parallel)
    logger.info("Step 1/3: Analyzing audio with Gemini...")
    ctx.invoke(gemini_audio, input_dir=input_dir, output_dir=output_dir, workers=workers)
    
    # Step 2: Generate reports
    logger.info("Step 2/3: Generating reports...")
    ctx.invoke(
        report,
        transcriptions=transcriptions_file,
        analysis=analysis_file,
        output=str(Path(output_dir) / "report.xlsx"),
        format='all',
        audio_dir=audio_dir
    )
    
    # Step 3: Generate visualizations
    logger.info("Step 3/3: Generating visualizations...")
    ctx.invoke(
        visualize,
        transcriptions=transcriptions_file,
        analysis=analysis_file,
        output_dir=charts_dir,
        audio_dir=audio_dir
    )
    
    logger.info(f"Gemini pipeline complete! All outputs saved to: {output_dir}")
    logger.info(f"  - Transcriptions: {transcriptions_file}")
    logger.info(f"  - Analysis: {analysis_file}")
    logger.info(f"  - Reports: {output_dir}/report.*")
    logger.info(f"  - Charts: {charts_dir}/")


@cli.command()
@click.option('--input-dir', required=True, help='Directory containing audio files')
@click.option('--output-dir', default='data/output', help='Output directory for all results')
@click.option('--audio-dir', help='Audio directory (if different from input-dir)')
@click.pass_context
def pipeline(ctx, input_dir, output_dir, audio_dir):
    """Run complete pipeline: transcribe, analyze, report, and visualize"""
    logger = ctx.obj['logger']
    logger.info("Starting full pipeline")
    
    if audio_dir is None:
        audio_dir = input_dir
    
    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # File paths
    transcriptions_file = str(Path(output_dir) / "transcriptions.json")
    analysis_file = str(Path(output_dir) / "analysis.json")
    charts_dir = str(Path(output_dir) / "charts")
    
    # Step 1: Transcribe
    logger.info("Step 1/4: Transcribing audio files...")
    ctx.invoke(transcribe, input_dir=input_dir, output=transcriptions_file)
    
    # Step 2: Analyze
    logger.info("Step 2/4: Analyzing transcriptions...")
    ctx.invoke(analyze, input=transcriptions_file, output=analysis_file)
    
    # Step 3: Generate reports
    logger.info("Step 3/4: Generating reports...")
    ctx.invoke(
        report,
        transcriptions=transcriptions_file,
        analysis=analysis_file,
        output=str(Path(output_dir) / "report.xlsx"),
        format='all',
        audio_dir=audio_dir
    )
    
    # Step 4: Generate visualizations
    logger.info("Step 4/4: Generating visualizations...")
    ctx.invoke(
        visualize,
        transcriptions=transcriptions_file,
        analysis=analysis_file,
        output_dir=charts_dir,
        audio_dir=audio_dir
    )
    
    logger.info(f"Pipeline complete! All outputs saved to: {output_dir}")
    logger.info(f"  - Transcriptions: {transcriptions_file}")
    logger.info(f"  - Analysis: {analysis_file}")
    logger.info(f"  - Reports: {output_dir}/report.*")
    logger.info(f"  - Charts: {charts_dir}/")


@cli.command()
@click.option('--transcriptions', required=True, help='Transcriptions JSON file')
@click.option('--analysis', required=True, help='Analysis JSON file')
@click.option('--audio-dir', help='Audio directory for duration extraction')
@click.pass_context
def import_to_db(ctx, transcriptions, analysis, audio_dir):
    """Import data into database"""
    logger = ctx.obj['logger']
    logger.info("Importing data to database")
    
    # Load data
    transcriptions_data = load_json(transcriptions)
    analysis_data = load_json(analysis)
    
    # Create database
    db = create_database(ctx.obj['config'])
    
    # Import calls
    for call_id, transcription in transcriptions_data.items():
        call_datetime = None
        duration = None
        
        # Try to extract datetime from call_id
        from src.utils import parse_call_id_datetime
        call_datetime = parse_call_id_datetime(call_id)
        
        # Try to get duration
        if audio_dir:
            try:
                import soundfile as sf
                import os
                audio_path = os.path.join(audio_dir, call_id)
                if os.path.exists(audio_path):
                    with sf.SoundFile(audio_path) as f:
                        duration = len(f) / f.samplerate
            except Exception as e:
                logger.debug(f"Could not get duration for {call_id}: {e}")
        
        db.insert_call(call_id, transcription, call_datetime, duration)
    
    # Import analysis
    for call_id, analysis_text in analysis_data.items():
        db.insert_analysis(call_id, analysis_text)
    
    # Get statistics
    stats = db.get_statistics()
    logger.info(f"Import complete!")
    logger.info(f"  - Total calls: {stats['total_calls']}")
    logger.info(f"  - Analyzed calls: {stats['analyzed_calls']}")


@cli.command()
@click.pass_context
def stats(ctx):
    """Show database statistics"""
    logger = ctx.obj['logger']
    
    # Create database
    db = create_database(ctx.obj['config'])
    
    # Get statistics
    stats = db.get_statistics()
    
    click.echo("\n=== DATABASE STATISTICS ===\n")
    click.echo(f"Total Calls: {stats['total_calls']}")
    click.echo(f"Analyzed Calls: {stats['analyzed_calls']}")
    
    if stats.get('sentiment_distribution'):
        click.echo("\nSentiment Distribution:")
        for sentiment, count in stats['sentiment_distribution'].items():
            click.echo(f"  - {sentiment}: {count}")
    
    if stats.get('resolution_distribution'):
        click.echo("\nResolution Distribution:")
        for status, count in stats['resolution_distribution'].items():
            click.echo(f"  - {status}: {count}")
    
    if stats.get('issue_categories'):
        click.echo("\nIssue Categories:")
        for category, count in stats['issue_categories'].items():
            click.echo(f"  - {category}: {count}")
    
    click.echo("")


if __name__ == '__main__':
    cli(obj={})
