"""
Visualization module for generating charts and graphs
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from wordcloud import WordCloud
from collections import Counter

from .utils import ensure_directory


logger = logging.getLogger(__name__)


class DataVisualizer:
    """
    Creates visualizations for call center analytics
    """
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        viz_config = self.config.get('visualization', {})
        
        # Set style
        plt.style.use(viz_config.get('style', 'ggplot'))
        
        # Set defaults
        self.figure_size = tuple(viz_config.get('figure_size', [12, 8]))
        self.dpi = viz_config.get('dpi', 150)
        self.color_palette = viz_config.get('color_palette', 'Set3')
    
    def _save_figure(self, output_path: str) -> None:
        """Save and close current figure"""
        ensure_directory(str(Path(output_path).parent))
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved visualization: {output_path}")
    
    def plot_call_duration_distribution(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """Plot distribution of call durations"""
        if 'Duration' not in df.columns or df['Duration'].isna().all():
            logger.warning("No duration data available")
            return
        
        plt.figure(figsize=(10, 6))
        durations = df['Duration'].dropna()
        
        # Use custom bins with more detail at lower values
        bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 900, 1200, 1800, 3000, 6000]
        bins = [b for b in bins if b <= durations.max()]
        if durations.max() not in bins:
            bins.append(int(durations.max()) + 100)
        
        plt.hist(durations, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(durations.mean(), color='red', linestyle='--', 
                   label=f'Mean: {durations.mean():.1f}s')
        plt.axvline(durations.median(), color='green', linestyle='--',
                   label=f'Median: {durations.median():.1f}s')
        
        plt.title('Call Duration Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Duration (seconds)', fontsize=12)
        plt.ylabel('Number of Calls', fontsize=12)
        plt.xscale('symlog', linthresh=600)  # Symmetric log scale with linear region up to 600s
        plt.xlim(left=0)  # Start x-axis at 0
        
        # Add more x-axis ticks
        max_duration = durations.max()
        tick_positions = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 900, 1200, 1800, 2400, 3000]
        tick_positions = [t for t in tick_positions if t <= max_duration]
        if max_duration > tick_positions[-1]:
            tick_positions.append(int(max_duration))
        plt.xticks(tick_positions, [str(int(t)) for t in tick_positions], rotation=45, ha='right')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        self._save_figure(output_path)
    
    def plot_resolution_status(
        self,
        df: pd.DataFrame,
        output_path: str,
        exclude_values: List[str] = None
    ) -> None:
        """Plot resolution status distribution"""
        if 'Resolution Status' not in df.columns:
            logger.warning("No resolution status data available")
            return
        
        # Filter data
        data = df['Resolution Status'].copy()
        # Always filter N/A
        data = data[~data.str.contains('N/A', case=False, na=False)]
        if exclude_values:
            data = data[~data.isin(exclude_values)]
        
        plt.figure(figsize=(10, 6))
        counts = data.value_counts()
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        bars = plt.bar(range(len(counts)), counts.values, color=colors, edgecolor='black')
        
        plt.title('Call Resolution Status Distribution', fontsize=16, fontweight='bold')
        plt.xlabel('Resolution Status', fontsize=12)
        plt.ylabel('Number of Calls', fontsize=12)
        plt.xticks(range(len(counts)), counts.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        self._save_figure(output_path)
    
    def plot_sentiment_trajectory(
        self,
        df: pd.DataFrame,
        output_path: str,
        exclude_values: List[str] = None
    ) -> None:
        """Plot sentiment trajectory distribution"""
        if 'Customer Sentiment Trajectory' not in df.columns:
            logger.warning("No sentiment data available")
            return
        
        # Filter data
        data = df['Customer Sentiment Trajectory'].copy()
        if exclude_values:
            data = data[~data.isin(exclude_values)]
        
        plt.figure(figsize=(10, 6))
        counts = data.value_counts()
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        # Pie chart
        plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
               colors=colors[:len(counts)], startangle=90,
               textprops={'fontsize': 11})
        
        plt.title('Customer Sentiment Trajectory Distribution', 
                 fontsize=16, fontweight='bold')
        
        self._save_figure(output_path)
    
    def plot_sentiment_vs_resolution(
        self,
        df: pd.DataFrame,
        output_path: str,
        exclude_values: List[str] = None
    ) -> None:
        """Plot sentiment trajectory vs resolution status heatmap"""
        if 'Customer Sentiment Trajectory' not in df.columns or 'Resolution Status' not in df.columns:
            logger.warning("Missing required columns for sentiment vs resolution")
            return
        
        # Filter data
        df_filtered = df.copy()
        if exclude_values:
            df_filtered = df_filtered[
                ~df_filtered['Customer Sentiment Trajectory'].isin(exclude_values) &
                ~df_filtered['Resolution Status'].isin(exclude_values)
            ]
        
        plt.figure(figsize=(12, 8))
        
        # Create crosstab
        ct = pd.crosstab(
            df_filtered['Customer Sentiment Trajectory'],
            df_filtered['Resolution Status']
        )
        
        # Heatmap
        sns.heatmap(ct, annot=True, fmt='d', cmap='YlGnBu', 
                   cbar_kws={'label': 'Number of Calls'},
                   linewidths=0.5, linecolor='gray')
        
        plt.title('Sentiment Trajectory vs Resolution Status', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Resolution Status', fontsize=12)
        plt.ylabel('Customer Sentiment Trajectory', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        self._save_figure(output_path)
    
    def plot_issue_frequency(
        self,
        df: pd.DataFrame,
        output_path: str,
        issue_columns: List[str] = None
    ) -> None:
        """Plot frequency of different issue categories"""
        if issue_columns is None:
            issue_columns = ['Customer Issues', 'Vehicle Issues', 'AS Issues', 'Other Issues']
        
        # Filter to existing columns
        existing_cols = [col for col in issue_columns if col in df.columns]
        
        if not existing_cols:
            logger.warning("No issue category columns found")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Count non-empty values in each category
        counts = {}
        for col in existing_cols:
            non_empty = df[col].astype(str).str.strip().replace('', pd.NA).notna().sum()
            counts[col] = non_empty
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']
        
        plt.barh(list(counts.keys()), list(counts.values()), 
                color=colors[:len(counts)], edgecolor='black')
        
        plt.title('Issue Category Frequency', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Calls', fontsize=12)
        plt.ylabel('Issue Category', fontsize=12)
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (category, count) in enumerate(counts.items()):
            plt.text(count, i, f' {count}', va='center', fontsize=10)
        
        self._save_figure(output_path)
    
    def plot_top_issues(
        self,
        df: pd.DataFrame,
        output_path: str,
        column: str,
        top_n: int = 5,
        title: str = None
    ) -> None:
        """Plot top N issues from a specific category"""
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Extract and count issues
        all_issues = (
            df[column]
            .dropna()
            .astype(str)
            .str.split(',')
            .explode()
            .str.strip()
        )
        all_issues = all_issues[all_issues != '']
        
        if len(all_issues) == 0:
            logger.warning(f"No issues found in {column}")
            return
        
        counts = all_issues.value_counts().head(top_n)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(counts)))
        plt.bar(range(len(counts)), counts.values, color=colors, edgecolor='black')
        
        plt.title(title or f'Top {top_n} {column}', fontsize=16, fontweight='bold')
        plt.ylabel('Count', fontsize=12)
        plt.xticks(range(len(counts)), counts.index, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(counts.values):
            plt.text(i, v, f'{v}', ha='center', va='bottom', fontsize=10)
        
        self._save_figure(output_path)
    
    def plot_issues_wordcloud(
        self,
        df: pd.DataFrame,
        output_path: str,
        column: str = 'Key Issues'
    ) -> None:
        """Generate word cloud from issues"""
        if column not in df.columns:
            logger.warning(f"Column {column} not found")
            return
        
        # Combine all issues
        all_issues = ' '.join(df[column].dropna().astype(str))
        
        if not all_issues.strip():
            logger.warning(f"No text found in {column}")
            return
        
        plt.figure(figsize=(12, 6))
        
        wordcloud = WordCloud(
            width=1200,
            height=600,
            background_color='white',
            colormap='viridis',
            collocations=False,
            max_words=100
        ).generate(all_issues)
        
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Common Customer Issues Word Cloud', 
                 fontsize=16, fontweight='bold', pad=20)
        
        self._save_figure(output_path)
    
    def plot_duration_vs_resolution(
        self,
        df: pd.DataFrame,
        output_path: str,
        sentiment_colored: bool = True
    ) -> None:
        """Plot call duration vs resolution status"""
        required_cols = ['Duration', 'Resolution Status']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for duration vs resolution")
            return
        
        df_plot = df.dropna(subset=['Duration', 'Resolution Status']).copy()
        
        # Filter out N/A resolution statuses
        df_plot = df_plot[~df_plot['Resolution Status'].str.contains('N/A', case=False, na=False)]
        
        if len(df_plot) == 0:
            logger.warning("No data available for duration vs resolution plot")
            return
        
        plt.figure(figsize=(12, 6))
        
        # Create categorical codes for x-axis
        df_plot['resolution_code'] = pd.Categorical(
            df_plot['Resolution Status']
        ).codes
        
        # Add jitter
        jitter = np.random.uniform(-0.2, 0.2, size=len(df_plot))
        x_jittered = df_plot['resolution_code'] + jitter
        
        # Color by sentiment if available and requested
        if sentiment_colored and 'Customer Sentiment Trajectory' in df.columns:
            # Normalize sentiment values (map old values to new ones)
            df_plot['Customer Sentiment Trajectory'] = df_plot['Customer Sentiment Trajectory'].replace({
                'consistently frustrated': 'consistently negative',
                'positive to negative': 'consistently negative'  # Map to negative if needed
            })
            
            # Define exact color mapping
            color_map = {
                'consistently neutral': '#FFFF99',      # Light yellow
                'consistently positive': '#90EE90',     # Light green
                'consistently negative': '#FFB6C1',     # Light red/pink
                'negative to positive': '#87CEEB'       # Light blue
            }
            
            for sentiment in ['consistently neutral', 'consistently positive', 'consistently negative', 'negative to positive']:
                mask = df_plot['Customer Sentiment Trajectory'] == sentiment
                if mask.any():
                    plt.scatter(
                        x_jittered[mask],
                        df_plot.loc[mask, 'Duration'],
                        c=color_map[sentiment],
                        label=sentiment,
                        alpha=0.6,
                        s=50,
                        edgecolors='black',
                        linewidths=0.5
                    )
            plt.legend(title='Sentiment', bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.scatter(x_jittered, df_plot['Duration'], alpha=0.6, s=50)
        
        plt.xticks(
            range(len(df_plot['Resolution Status'].unique())),
            df_plot['Resolution Status'].unique(),
            rotation=45,
            ha='right'
        )
        
        plt.xlabel('Resolution Status', fontsize=12)
        plt.ylabel('Call Duration (seconds)', fontsize=12)
        plt.title('Call Duration vs Resolution Status', fontsize=16, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        
        self._save_figure(output_path)
    
    def plot_issue_categories_distribution(
        self,
        df: pd.DataFrame,
        output_path: str
    ) -> None:
        """Plot distribution of categorized issues from key_issues field"""
        if 'Key Issues' not in df.columns:
            logger.warning("No key_issues data available")
            return
        
        # Extract all categories from key_issues lists
        all_categories = []
        for issues in df['Key Issues'].dropna():
            # Issues are pipe-separated strings from reporting.py
            # e.g., "Navigation malfunction | Auto-hold malfunction (won't engage or release)"
            if isinstance(issues, str) and issues != "N/A":
                # Split by pipe and trim whitespace
                categories = [cat.strip() for cat in issues.split('|') if cat.strip()]
                all_categories.extend(categories)
            elif isinstance(issues, list):
                all_categories.extend([str(cat).strip() for cat in issues if str(cat).strip()])
        
        if not all_categories:
            logger.warning("No valid categories found in key_issues")
            return
        
        # Count categories
        category_counts = Counter(all_categories)
        
        # Filter out generic/non-informative categories
        exclude = ['Confirmation needed', 'N/A', 'Unclear from transcript']
        category_counts = {k: v for k, v in category_counts.items() if k not in exclude}
        
        if not category_counts:
            logger.warning("No valid categories after filtering")
            return
        
        # Get top categories for pie chart
        top_n = 12
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_categories) > top_n:
            top_categories = dict(sorted_categories[:top_n])
            other_count = sum(count for _, count in sorted_categories[top_n:])
            if other_count > 0:
                top_categories['Other'] = other_count
        else:
            top_categories = dict(sorted_categories)
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.Set3(np.linspace(0, 1, len(top_categories)))
        
        wedges, texts, autotexts = plt.pie(
            top_categories.values(),
            labels=top_categories.keys(),
            autopct='%1.1f%%',
            colors=colors,
            startangle=90,
            textprops={'fontsize': 9}
        )
        
        # Make percentage text bold
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        plt.title(f'Issue Categories Distribution (Top {len(top_categories)} Categories)\nTotal: {sum(category_counts.values())} issues from {len(df)} calls', 
                 fontsize=14, fontweight='bold')
        
        self._save_figure(output_path)
    
    def plot_top_issue_categories(
        self,
        df: pd.DataFrame,
        output_path: str,
        top_n: int = 15
    ) -> None:
        """Plot horizontal bar chart of top issue categories"""
        if 'Key Issues' not in df.columns:
            logger.warning("No key_issues data available")
            return
        
        # Extract all categories
        all_categories = []
        for issues in df['Key Issues'].dropna():
            # Issues are pipe-separated strings from reporting.py
            # e.g., "Navigation malfunction | Auto-hold malfunction (won't engage or release)"
            if isinstance(issues, str) and issues != "N/A":
                # Split by pipe and trim whitespace
                categories = [cat.strip() for cat in issues.split('|') if cat.strip()]
                all_categories.extend(categories)
            elif isinstance(issues, list):
                all_categories.extend([str(cat).strip() for cat in issues if str(cat).strip()])
        
        if not all_categories:
            logger.warning("No valid categories found")
            return
        
        # Count and filter
        category_counts = Counter(all_categories)
        exclude = ['Confirmation needed', 'N/A', 'Unclear from transcript']
        category_counts = {k: v for k, v in category_counts.items() if k not in exclude}
        
        # Get top N
        sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        categories = [cat for cat, _ in sorted_categories]
        counts = [count for _, count in sorted_categories]
        
        plt.figure(figsize=(12, 8))
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(categories)))
        
        y_pos = np.arange(len(categories))
        bars = plt.barh(y_pos, counts, color=colors, edgecolor='black')
        
        plt.yticks(y_pos, categories, fontsize=10)
        plt.xlabel('Number of Occurrences', fontsize=12)
        plt.title(f'Top {top_n} Issue Categories\nTotal Calls: {len(df)}', 
                 fontsize=14, fontweight='bold')
        plt.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, count) in enumerate(zip(bars, counts)):
            plt.text(count + 0.5, i, str(count), 
                    va='center', fontsize=9, fontweight='bold')
        
        plt.gca().invert_yaxis()  # Highest at top
        
        self._save_figure(output_path)
    
    def generate_all_visualizations(
        self,
        df: pd.DataFrame,
        output_dir: str,
        exclude_values: List[str] = None
    ) -> Dict[str, str]:
        """
        Generate all standard visualizations
        
        Args:
            df: DataFrame with analysis data
            output_dir: Directory to save visualizations
            exclude_values: Values to exclude from certain plots
            
        Returns:
            Dictionary mapping chart name to file path
        """
        ensure_directory(output_dir)
        
        if exclude_values is None:
            exclude_values = [
                'Unclear from transcript',
                'N/A',
                'consistently confused',
                'consistently inquisitive'
            ]
        
        outputs = {}
        
        # 1. Call Duration Distribution
        path = str(Path(output_dir) / "call_duration_distribution.png")
        self.plot_call_duration_distribution(df, path)
        outputs['duration_dist'] = path
        
        # 2. Resolution Status
        path = str(Path(output_dir) / "resolution_status.png")
        self.plot_resolution_status(df, path, exclude_values)
        outputs['resolution'] = path
        
        # 3. Sentiment Trajectory
        path = str(Path(output_dir) / "sentiment_trajectory.png")
        self.plot_sentiment_trajectory(df, path, exclude_values)
        outputs['sentiment'] = path
        
        # 4. Sentiment vs Resolution
        path = str(Path(output_dir) / "sentiment_vs_resolution.png")
        self.plot_sentiment_vs_resolution(df, path, exclude_values)
        outputs['sentiment_resolution'] = path
        
        # 5. Issue Frequency (legacy - skipped, using top_issue_categories instead)
        # path = str(Path(output_dir) / "issue_frequency.png")
        # self.plot_issue_frequency(df, path)
        # outputs['issue_freq'] = path
        
        # 6. Issue Categories Distribution (from key_issues)
        path = str(Path(output_dir) / "issue_categories_distribution.png")
        self.plot_issue_categories_distribution(df, path)
        outputs['issue_categories'] = path
        
        # 7. Top Issue Categories
        path = str(Path(output_dir) / "top_issue_categories.png")
        self.plot_top_issue_categories(df, path, top_n=15)
        outputs['top_categories'] = path
        
        # 8. Top Issues by Category
        for category in ['Customer Issues', 'Vehicle Issues', 'AS Issues', 'Other Issues']:
            if category in df.columns:
                path = str(Path(output_dir) / f"top_{category.lower().replace(' ', '_')}.png")
                self.plot_top_issues(df, path, category, top_n=5, 
                                   title=f'Top 5 {category}')
                outputs[f'top_{category}'] = path
        
        # 9. Issues Word Cloud
        path = str(Path(output_dir) / "issues_wordcloud.png")
        self.plot_issues_wordcloud(df, path)
        outputs['wordcloud'] = path
        
        # 10. Duration vs Resolution
        path = str(Path(output_dir) / "duration_vs_resolution.png")
        self.plot_duration_vs_resolution(df, path)
        outputs['duration_resolution'] = path
        
        logger.info(f"Generated {len(outputs)} visualizations in {output_dir}")
        
        return outputs
