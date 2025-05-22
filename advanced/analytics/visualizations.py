"""
Visualization functions for the advanced DuckDB implementation.

This module provides functions to create visualizations from the UK AI jobs data,
including charts, graphs, and other visual representations.
"""

import os
import logging
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA

from advanced.config import ANALYTICS_CONFIG
from advanced.analytics.metrics import (
    calculate_ai_impact_distribution,
    calculate_top_companies,
    calculate_skills_importance,
    perform_cluster_analysis
)

# Set up logging
logger = logging.getLogger(__name__)

# Configure matplotlib
plt.style.use('ggplot')
sns.set_palette("viridis")


def save_figure(fig: plt.Figure, filename: str) -> str:
    """
    Save a matplotlib figure to the visualization directory.
    
    Args:
        fig: Matplotlib figure to save
        filename: Name of the file (without path)
        
    Returns:
        Full path to the saved file
    """
    viz_dir = ANALYTICS_CONFIG.get("visualization_dir")
    os.makedirs(viz_dir, exist_ok=True)
    
    filepath = os.path.join(viz_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    
    logger.info(f"Saved visualization to {filepath}")
    return filepath


def plot_ai_impact_distribution(df: pd.DataFrame) -> str:
    """
    Plot the distribution of AI impact scores.
    
    Args:
        df: DataFrame with AI impact distribution data
        
    Returns:
        Path to the saved visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Extract data
    categories = ['low_impact_count', 'medium_impact_count', 'high_impact_count', 'transformative_impact_count']
    category_labels = ['Low', 'Medium', 'High', 'Transformative']
    values = [df[cat].iloc[0] for cat in categories]
    
    # Create bar chart
    ax1.bar(category_labels, values, color=sns.color_palette("viridis", len(categories)))
    ax1.set_title('AI Impact Category Distribution')
    ax1.set_ylabel('Number of Jobs')
    ax1.set_xlabel('AI Impact Category')
    
    # Create pie chart
    ax2.pie(values, labels=category_labels, autopct='%1.1f%%', 
            colors=sns.color_palette("viridis", len(categories)))
    ax2.set_title('AI Impact Categories')
    
    # Add statistics
    stats_text = (
        f"Total Jobs: {df['total_jobs'].iloc[0]}\n"
        f"Average Impact: {df['avg_impact'].iloc[0]:.2f}\n"
        f"Median Impact: {df['median_impact'].iloc[0]:.2f}\n"
        f"Min/Max: {df['min_impact'].iloc[0]:.2f}/{df['max_impact'].iloc[0]:.2f}"
    )
    fig.text(0.5, 0.01, stats_text, ha='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.tight_layout()
    
    return save_figure(fig, 'ai_impact_distribution.png')


def plot_top_companies(df: pd.DataFrame) -> str:
    """
    Plot the top companies with the highest number of AI-impacted jobs.
    
    Args:
        df: DataFrame with top companies data
        
    Returns:
        Path to the saved visualization
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal bar chart
    companies = df['company']
    job_counts = df['job_count']
    avg_impacts = df['avg_ai_impact']
    
    # Color bars by average AI impact
    norm = plt.Normalize(avg_impacts.min(), avg_impacts.max())
    colors = cm.viridis(norm(avg_impacts))
    
    bars = ax.barh(companies, job_counts, color=colors)
    
    # Add average AI impact as text
    for i, (company, count, impact) in enumerate(zip(companies, job_counts, avg_impacts)):
        ax.text(count + 0.1, i, f"AI: {impact:.2f}", va='center')
    
    # Add a colorbar
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Average AI Impact')
    
    ax.set_title('Top Companies by AI Job Count')
    ax.set_xlabel('Number of Jobs')
    ax.set_ylabel('Company')
    
    plt.tight_layout()
    
    return save_figure(fig, 'top_companies.png')


def plot_skills_heatmap(df: pd.DataFrame) -> str:
    """
    Plot a heatmap of skills importance.
    
    Args:
        df: DataFrame with skills importance data
        
    Returns:
        Path to the saved visualization
    """
    # Prepare data for heatmap
    pivot_df = df.pivot_table(
        index='skill_name', 
        values=['job_count', 'avg_ai_impact', 'required_count'],
        aggfunc='first'
    ).sort_values('job_count', ascending=False)
    
    # Normalize values for better visualization
    for col in pivot_df.columns:
        pivot_df[col] = (pivot_df[col] - pivot_df[col].min()) / (pivot_df[col].max() - pivot_df[col].min())
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create heatmap
    sns.heatmap(pivot_df, cmap="viridis", annot=False, ax=ax)
    
    ax.set_title('Skills Importance Heatmap (Normalized Values)')
    
    plt.tight_layout()
    
    return save_figure(fig, 'skills_heatmap.png')


def plot_cluster_analysis(df_with_clusters: pd.DataFrame, cluster_stats: pd.DataFrame) -> str:
    """
    Plot the results of cluster analysis.
    
    Args:
        df_with_clusters: DataFrame with cluster assignments
        cluster_stats: DataFrame with cluster statistics
        
    Returns:
        Path to the saved visualization
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    
    # Extract cluster data
    n_clusters = len(cluster_stats)
    clusters = sorted(df_with_clusters['cluster'].unique())
    
    # Apply PCA to visualize clusters
    features = ['ai_impact', 'salary_min', 'salary_max', 'remote_percentage']
    data_for_pca = df_with_clusters[features].fillna(0)
    
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(data_for_pca)
    
    # Add PCA results to dataframe
    df_with_clusters['pca_x'] = pca_result[:, 0]
    df_with_clusters['pca_y'] = pca_result[:, 1]
    
    # Plot clusters
    colors = sns.color_palette("viridis", n_clusters)
    
    for cluster, color in zip(clusters, colors):
        mask = df_with_clusters['cluster'] == cluster
        ax1.scatter(
            df_with_clusters.loc[mask, 'pca_x'], 
            df_with_clusters.loc[mask, 'pca_y'],
            c=[color], 
            label=f'Cluster {cluster}'
        )
    
    ax1.set_title('Job Clusters (PCA Visualization)')
    ax1.set_xlabel('Principal Component 1')
    ax1.set_ylabel('Principal Component 2')
    ax1.legend()
    
    # Plot cluster statistics
    cluster_sizes = cluster_stats[('ai_impact', 'count')].values
    ai_impacts = cluster_stats[('ai_impact', 'mean')].values
    salaries = cluster_stats[('salary_min', 'mean')].values
    
    # Normalize salaries for bubble size
    size_norm = (salaries - salaries.min()) / (salaries.max() - salaries.min()) * 1000 + 100
    
    # Create bubble chart
    for i, (cluster, size, impact, salary) in enumerate(zip(clusters, cluster_sizes, ai_impacts, salaries)):
        ax2.scatter(
            impact, 
            salary,
            s=size, 
            c=[colors[i]], 
            alpha=0.7,
            label=f'Cluster {cluster}'
        )
        ax2.text(impact, salary, f'{cluster}', ha='center', va='center')
    
    ax2.set_title('Cluster Characteristics')
    ax2.set_xlabel('Average AI Impact')
    ax2.set_ylabel('Average Minimum Salary')
    
    plt.tight_layout()
    
    return save_figure(fig, 'cluster_analysis.png')


def create_visualizations(db: JobsDatabase) -> List[str]:
    """
    Create all visualizations from the database.
    
    Args:
        db: JobsDatabase instance
        
    Returns:
        List of paths to saved visualizations
    """
    paths = []
    
    try:
        # AI impact distribution
        logger.info("Creating AI impact distribution visualization...")
        df_impact = calculate_ai_impact_distribution(db)
        path = plot_ai_impact_distribution(df_impact)
        paths.append(path)
        
        # Top companies
        logger.info("Creating top companies visualization...")
        df_companies = calculate_top_companies(db)
        path = plot_top_companies(df_companies)
        paths.append(path)
        
        # Skills heatmap
        logger.info("Creating skills heatmap visualization...")
        df_skills = calculate_skills_importance(db)
        path = plot_skills_heatmap(df_skills)
        paths.append(path)
        
        # Cluster analysis
        logger.info("Creating cluster analysis visualization...")
        df_with_clusters, cluster_stats = perform_cluster_analysis(db)
        path = plot_cluster_analysis(df_with_clusters, cluster_stats)
        paths.append(path)
        
        logger.info(f"Created {len(paths)} visualizations")
    except Exception as e:
        logger.error(f"Failed to create visualizations: {e}")
    
    return paths