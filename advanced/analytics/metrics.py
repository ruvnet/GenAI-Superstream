"""
Analytics metrics calculation for the advanced DuckDB implementation.

This module provides functions to calculate various metrics and statistics from
the job data stored in the database.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA

from advanced.db.queries import JobsDatabase
from advanced.config import ANALYTICS_CONFIG

# Set up logging
logger = logging.getLogger(__name__)


def calculate_ai_impact_distribution(db: JobsDatabase) -> pd.DataFrame:
    """
    Calculate the distribution of AI impact scores.
    
    Args:
        db: JobsDatabase instance
        
    Returns:
        DataFrame with AI impact distribution statistics
    """
    query = """
    SELECT 
        COUNT(*) as total_jobs,
        AVG(ai_impact) as avg_impact,
        MIN(ai_impact) as min_impact,
        MAX(ai_impact) as max_impact,
        STDDEV(ai_impact) as std_impact,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY ai_impact) as q1_impact,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY ai_impact) as median_impact,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY ai_impact) as q3_impact,
        COUNT(CASE WHEN ai_impact_category = 'low' THEN 1 END) as low_impact_count,
        COUNT(CASE WHEN ai_impact_category = 'medium' THEN 1 END) as medium_impact_count,
        COUNT(CASE WHEN ai_impact_category = 'high' THEN 1 END) as high_impact_count,
        COUNT(CASE WHEN ai_impact_category = 'transformative' THEN 1 END) as transformative_impact_count
    FROM job_postings
    WHERE ai_impact IS NOT NULL
    """
    return db.to_dataframe(query)


def calculate_top_companies(db: JobsDatabase, min_ai_impact: float = 0.5, limit: int = 10) -> pd.DataFrame:
    """
    Calculate the top companies with the highest number of AI-impacted jobs.
    
    Args:
        db: JobsDatabase instance
        min_ai_impact: Minimum AI impact score to consider
        limit: Maximum number of companies to return
        
    Returns:
        DataFrame with companies and their job counts
    """
    query = f"""
    SELECT 
        company,
        COUNT(*) as job_count,
        AVG(ai_impact) as avg_ai_impact,
        AVG(CASE WHEN salary_min IS NOT NULL THEN salary_min ELSE NULL END) as avg_min_salary,
        AVG(CASE WHEN salary_max IS NOT NULL THEN salary_max ELSE NULL END) as avg_max_salary,
        COUNT(DISTINCT ai_impact_category) as impact_category_diversity
    FROM job_postings
    WHERE ai_impact > {min_ai_impact}
    GROUP BY company
    ORDER BY job_count DESC, avg_ai_impact DESC
    LIMIT {limit}
    """
    return db.to_dataframe(query)


def calculate_skills_importance(db: JobsDatabase, top_n: int = 20) -> pd.DataFrame:
    """
    Calculate the importance of different skills based on frequency and AI impact.
    
    Args:
        db: JobsDatabase instance
        top_n: Number of top skills to return
        
    Returns:
        DataFrame with skills importance analysis
    """
    query = f"""
    SELECT 
        s.skill_name,
        s.skill_category,
        COUNT(*) as job_count,
        AVG(j.ai_impact) as avg_ai_impact,
        SUM(CASE WHEN s.is_required = TRUE THEN 1 ELSE 0 END) as required_count,
        AVG(CASE WHEN j.salary_min IS NOT NULL THEN j.salary_min ELSE NULL END) as avg_min_salary
    FROM job_skills s
    JOIN job_postings j ON s.job_id = j.job_id
    GROUP BY s.skill_name, s.skill_category
    ORDER BY job_count DESC, avg_ai_impact DESC
    LIMIT {top_n}
    """
    return db.to_dataframe(query)


def perform_cluster_analysis(db: JobsDatabase, n_clusters: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform cluster analysis on job data to identify patterns.
    
    Args:
        db: JobsDatabase instance
        n_clusters: Number of clusters to identify
        
    Returns:
        Tuple of (DataFrame with cluster assignments, DataFrame with cluster statistics)
    """
    # Get job data
    query = """
    SELECT job_id, title, ai_impact, salary_min, salary_max, remote_percentage,
           date_posted, company, location
    FROM job_postings
    WHERE ai_impact IS NOT NULL
    """
    df = db.to_dataframe(query)
    
    # Preprocess data for clustering
    features = preprocess_data_for_ml(df)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Add cluster assignments to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Calculate cluster statistics
    cluster_stats = df_with_clusters.groupby('cluster').agg({
        'ai_impact': ['mean', 'std', 'count'],
        'salary_min': ['mean', 'std'],
        'salary_max': ['mean', 'std'],
        'remote_percentage': ['mean'],
        'title': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'Unknown'
    })
    
    return df_with_clusters, cluster_stats


def preprocess_data_for_ml(df: pd.DataFrame) -> np.ndarray:
    """
    Preprocess job data for machine learning.
    
    Args:
        df: DataFrame containing job data
        
    Returns:
        Numpy array of features for machine learning
    """
    # Convert date_posted to days since posting
    df['date_posted'] = pd.to_datetime(df['date_posted'])
    now = pd.Timestamp.now()
    df['days_since_posted'] = (now - df['date_posted']).dt.days
    
    # Create feature vectors from job titles using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=20)
    title_features = tfidf.fit_transform(df['title'].fillna(''))
    
    # Create a feature matrix
    feature_df = pd.DataFrame(title_features.toarray(), 
                            columns=[f'title_{i}' for i in range(title_features.shape[1])])
    
    # Add numeric features
    feature_df['ai_impact'] = df['ai_impact'].values
    feature_df['salary_min'] = df['salary_min'].fillna(df['salary_min'].mean()).values
    feature_df['salary_max'] = df['salary_max'].fillna(df['salary_max'].mean()).values
    feature_df['remote_percentage'] = df['remote_percentage'].fillna(0).values
    feature_df['days_since_posted'] = df['days_since_posted'].values
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)
    
    return scaled_features