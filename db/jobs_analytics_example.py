#!/usr/bin/env python3
"""
Example script demonstrating how to use the UK jobs database with scikit-learn for analytics.
This script shows how to:
1. Load data from the DuckDB database
2. Preprocess the data for machine learning
3. Run simple clustering and trend analysis on AI impact
4. Visualize the results
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Import the JobsDatabase class from init_duckdb.py
from init_duckdb import JobsDatabase

def generate_sample_data(db, num_samples=100):
    """
    Generate sample job data for demonstration purposes.
    In a real scenario, this would be replaced with actual scraped job data.
    
    Args:
        db: JobsDatabase instance
        num_samples: Number of sample jobs to generate
    """
    import random
    from datetime import datetime, timedelta
    
    job_titles = [
        "Data Scientist", "Machine Learning Engineer", "AI Research Scientist",
        "Data Engineer", "DevOps Engineer", "Software Developer", "Frontend Developer",
        "Backend Developer", "Full Stack Developer", "Product Manager", "UX Designer",
        "Cloud Architect", "Systems Engineer", "Network Engineer", "Security Analyst",
        "Business Analyst", "Project Manager", "Scrum Master", "Technical Writer"
    ]
    
    companies = [
        "TechCorp UK", "AI Solutions Ltd", "Data Insights Co", "Cloud Services UK",
        "Innovative Systems", "Digital Transformers", "Tech Innovate", "Future Tech",
        "Smart Solutions", "Code Masters", "Web Experts", "Mobile Innovators",
        "Enterprise Solutions", "Security Experts", "Analytics Pro"
    ]
    
    locations = [
        "London, UK", "Manchester, UK", "Birmingham, UK", "Edinburgh, UK",
        "Glasgow, UK", "Liverpool, UK", "Leeds, UK", "Bristol, UK",
        "Cardiff, UK", "Belfast, UK", "Newcastle, UK", "Sheffield, UK"
    ]
    
    descriptions = [
        "Join our team to work on cutting-edge AI solutions for enterprise clients...",
        "We are looking for a talented professional to help build our next generation platform...",
        "Work with a team of experts to develop innovative solutions using the latest technologies...",
        "Help transform our data infrastructure and build scalable systems for analytics...",
        "Join our growing team to develop and maintain critical business applications...",
        "Work on challenging problems and help define the future of our technical strategy..."
    ]
    
    sources = ["company_website", "linkedin", "indeed", "glassdoor", "reed"]
    
    # Different AI impact distributions for different job titles
    high_ai_impact = ["Data Scientist", "Machine Learning Engineer", "AI Research Scientist"]
    medium_ai_impact = ["Data Engineer", "Cloud Architect", "Backend Developer"]
    
    # Generate random job data
    jobs = []
    for i in range(num_samples):
        title = random.choice(job_titles)
        
        # Assign AI impact based on job title category
        if title in high_ai_impact:
            ai_impact = random.uniform(0.7, 0.95)
        elif title in medium_ai_impact:
            ai_impact = random.uniform(0.4, 0.7)
        else:
            ai_impact = random.uniform(0.1, 0.4)
        
        # Generate random date within the last year
        days_ago = random.randint(1, 365)
        date_posted = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        # Generate random salary range based on job title
        base_salary = 40000
        if title in high_ai_impact:
            salary_min = base_salary + random.randint(20000, 30000)
        elif title in medium_ai_impact:
            salary_min = base_salary + random.randint(10000, 20000)
        else:
            salary_min = base_salary + random.randint(0, 10000)
        
        salary_max = salary_min + random.randint(5000, 20000)
        salary = f"£{salary_min} - £{salary_max}"
        
        job = {
            "job_id": f"JOB-2023-{1000 + i}",
            "title": title,
            "company": random.choice(companies),
            "location": random.choice(locations),
            "salary": salary,
            "description": random.choice(descriptions),
            "ai_impact": ai_impact,
            "date_posted": date_posted,
            "source": random.choice(sources)
        }
        jobs.append(job)
    
    # Insert the generated jobs into the database
    db.insert_jobs_batch(jobs)
    print(f"Generated and inserted {num_samples} sample jobs")


def preprocess_data_for_ml(df):
    """
    Preprocess the job data for machine learning.
    
    Args:
        df: pandas DataFrame containing job data
        
    Returns:
        Preprocessed DataFrame and feature vectors
    """
    # Extract salary range midpoints
    df['salary_text'] = df['salary'].fillna('£0 - £0')
    
    # Function to extract midpoint of salary range
    def extract_salary_midpoint(salary_text):
        try:
            # Extract numbers from salary text
            salary_parts = salary_text.replace('£', '').replace(',', '').split('-')
            if len(salary_parts) == 2:
                min_salary = float(salary_parts[0].strip())
                max_salary = float(salary_parts[1].strip())
                return (min_salary + max_salary) / 2
            return 0
        except:
            return 0
    
    df['salary_midpoint'] = df['salary_text'].apply(extract_salary_midpoint)
    
    # Create feature vectors from job titles using TF-IDF
    tfidf = TfidfVectorizer(stop_words='english', max_features=100)
    title_features = tfidf.fit_transform(df['title'])
    
    # Create a feature matrix with numeric columns
    feature_df = pd.DataFrame(title_features.toarray(), 
                             columns=[f'title_feature_{i}' for i in range(title_features.shape[1])])
    
    # Add other numeric features
    feature_df['ai_impact'] = df['ai_impact'].values
    feature_df['salary_midpoint'] = df['salary_midpoint'].values
    
    # Add days since posting
    df['date_posted'] = pd.to_datetime(df['date_posted'])
    now = pd.Timestamp.now()
    df['days_since_posted'] = (now - df['date_posted']).dt.days
    feature_df['days_since_posted'] = df['days_since_posted'].values
    
    # Scale the features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_df)
    
    return df, scaled_features, feature_df.columns.tolist()


def cluster_analysis(df, features):
    """
    Perform cluster analysis on job data.
    
    Args:
        df: pandas DataFrame containing job data
        features: Feature matrix for clustering
        
    Returns:
        DataFrame with cluster assignments
    """
    # Determine optimal number of clusters (simplified for example)
    n_clusters = 4
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features)
    
    # Add cluster assignments to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = clusters
    
    # Analyze clusters
    cluster_stats = df_with_clusters.groupby('cluster').agg({
        'ai_impact': ['mean', 'std', 'count'],
        'salary_midpoint': ['mean', 'std'],
        'title': lambda x: pd.Series.mode(x)[0] if not pd.Series.mode(x).empty else 'Unknown'
    })
    
    print("\nCluster Analysis:")
    print(cluster_stats)
    
    return df_with_clusters


def visualize_results(df_with_clusters, features, feature_names):
    """
    Visualize the clustering results and AI impact trends.
    
    Args:
        df_with_clusters: DataFrame with cluster assignments
        features: Feature matrix
        feature_names: List of feature names
    """
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. PCA visualization of clusters
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(features)
    
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    for cluster in range(len(df_with_clusters['cluster'].unique())):
        mask = df_with_clusters['cluster'] == cluster
        axes[0].scatter(pca_result[mask, 0], pca_result[mask, 1], 
                     c=colors[cluster % len(colors)], label=f'Cluster {cluster}')
    
    axes[0].set_title('Job Clusters (PCA Visualization)')
    axes[0].set_xlabel('Principal Component 1')
    axes[0].set_ylabel('Principal Component 2')
    axes[0].legend()
    
    # 2. AI impact by job category
    ai_impact_by_title = df_with_clusters.groupby('title')['ai_impact'].mean().sort_values(ascending=False)
    top_titles = ai_impact_by_title.head(10)
    
    top_titles.plot(kind='barh', ax=axes[1])
    axes[1].set_title('Average AI Impact by Job Title')
    axes[1].set_xlabel('Average AI Impact Score')
    
    plt.tight_layout()
    plt.savefig('job_analytics_results.png')
    print("\nVisualization saved as 'job_analytics_results.png'")


def main():
    """
    Main function to demonstrate analytics on the UK jobs database.
    """
    # Initialize the database connection
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uk_jobs.duckdb")
    db = JobsDatabase(db_path)
    
    try:
        # Check if we need to generate sample data
        if len(db.query_jobs("SELECT COUNT(*) FROM job_postings")[0].values()) < 5:
            generate_sample_data(db, num_samples=100)
        
        # Export all data to a DataFrame
        df = db.to_dataframe()
        print(f"Loaded {len(df)} jobs from the database")
        
        # Display the distribution of AI impact
        ai_impact_distribution = db.get_ai_impact_distribution()
        print("\nAI Impact Distribution:")
        print(ai_impact_distribution)
        
        # Display top companies by AI jobs
        top_companies = db.get_top_companies_by_ai_jobs()
        print("\nTop Companies by AI-Impacted Jobs:")
        print(top_companies)
        
        # Preprocess data for ML
        df, features, feature_names = preprocess_data_for_ml(df)
        
        # Perform cluster analysis
        df_with_clusters = cluster_analysis(df, features)
        
        # Visualize results
        visualize_results(df_with_clusters, features, feature_names)
        
    except Exception as e:
        print(f"Error in analysis: {e}")
    finally:
        db.close()


if __name__ == "__main__":
    main()