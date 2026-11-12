"""
Exploratory Data Analysis (EDA) Module
Generates comprehensive distribution charts, statistical analyses, and summary figures
for the TED Talks dataset.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set aesthetics
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def run_eda(
    clean_data_path: str = "/workspace/data/processed/ted_clean.csv",
    output_dir: str = "/workspace/docs/assets"
):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(clean_data_path)
    print(f"Loaded {len(df)} records for EDA.")

    # 1. Talk Duration Distribution
    plt.figure(figsize=(9, 5))
    sns.histplot(df['duration_min'], bins=35, kde=True, color='#2563EB', edgecolor='white', alpha=0.85)
    plt.axvline(df['duration_min'].median(), color='#DC2626', linestyle='--', linewidth=2,
                label=f'Median: {df["duration_min"].median():.1f} min')
    plt.title('Distribution of TED Talk Durations (Minutes)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Duration (Minutes)', fontsize=11)
    plt.ylabel('Number of Talks', fontsize=11)
    plt.xlim(0, 45)
    plt.legend(frameon=True, facecolor='white')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_duration_distribution.png'), dpi=200)
    plt.close()

    # 2. Top 20 Tags
    all_tags = []
    for tags_str in df['tag_string']:
        if isinstance(tags_str, str):
            all_tags.extend(tags_str.split())
    tag_counts = pd.Series(all_tags).value_counts().head(20)

    plt.figure(figsize=(10, 6))
    palette = sns.color_palette("Blues_r", n_colors=20)
    bars = plt.barh(tag_counts.index[::-1], tag_counts.values[::-1], color=palette[::-1], edgecolor='white', height=0.7)
    plt.title('Top 20 Most Frequent TED Talk Topics / Tags', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Number of Talks', fontsize=11)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 10, bar.get_y() + bar.get_height() / 2, f'{int(w)}', va='center', fontsize=9, color='#333333')
    plt.xlim(0, max(tag_counts.values) * 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_top_tags.png'), dpi=200)
    plt.close()

    # 3. Top Emotional Ratings Distribution
    plt.figure(figsize=(10, 5))
    emotion_cols = [c for c in df.columns if c.startswith('emo_')]
    mean_emotions = df[emotion_cols].mean().sort_values(ascending=False)
    emotion_labels = [c.replace('emo_', '').capitalize() for c in mean_emotions.index]

    colors = ['#10B981' if v > 0.08 else '#3B82F6' if v > 0.04 else '#94A3B8' for v in mean_emotions.values]
    plt.bar(emotion_labels, mean_emotions.values, color=colors, edgecolor='white', width=0.65)
    plt.title('Average Emotional Response Distribution Across TED Talks', fontsize=14, fontweight='bold', pad=12)
    plt.ylabel('Average Normalized Share', fontsize=11)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_emotions.png'), dpi=200)
    plt.close()

    # 4. Views vs Comments (Log-Log Scale)
    plt.figure(figsize=(9, 5))
    plt.scatter(df['views'], df['comments'], alpha=0.4, color='#6366F1', edgecolors='none', s=25)
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Engagement Correlation: Views vs Comments (Log Scale)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Views (log scale)', fontsize=11)
    plt.ylabel('Comments (log scale)', fontsize=11)

    # Correlation coefficient
    valid = df[['views', 'comments']].dropna()
    corr = np.corrcoef(np.log10(valid['views']), np.log10(valid['comments']))[0, 1]
    plt.text(0.05, 0.90, f'Pearson r (log): {corr:.3f}', transform=plt.gca().transAxes,
             fontsize=11, fontweight='bold', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#ccc'))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_views_vs_comments.png'), dpi=200)
    plt.close()

    # 5. Top 10 Most Viewed TED Talks
    top_talks = df.sort_values(by='views', ascending=False).head(10)
    plt.figure(figsize=(11, 6))
    labels = [f"{row['main_speaker']}: {row['title'][:32]}..." if len(row['title']) > 32 else f"{row['main_speaker']}: {row['title']}"
              for _, row in top_talks.iterrows()]
    views_millions = top_talks['views'] / 1e6
    bars = plt.barh(labels[::-1], views_millions[::-1], color='#059669', edgecolor='white', height=0.65)
    plt.title('Top 10 Most Viewed TED Talks (Millions of Views)', fontsize=14, fontweight='bold', pad=12)
    plt.xlabel('Views (Millions)', fontsize=11)
    for bar in bars:
        w = bar.get_width()
        plt.text(w + 0.8, bar.get_y() + bar.get_height() / 2, f'{w:.1f}M', va='center', fontsize=9, color='#1F2937', fontweight='bold')
    plt.xlim(0, max(views_millions) * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eda_top_viewed_talks.png'), dpi=200)
    plt.close()

    print("All EDA charts generated successfully in", output_dir)


if __name__ == "__main__":
    run_eda()
