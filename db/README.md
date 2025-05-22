# UK Jobs Analytics Database with DuckDB

This module provides a database solution for analyzing UK jobs data with a focus on AI's impact on technical roles.

## Overview

The database is built using DuckDB, a high-performance analytical database system designed for fast queries on structured data. This solution is optimized for analytics workloads and seamless integration with pandas and scikit-learn for machine learning applications.

## Features

- **Efficient Storage**: Stores job postings with all relevant metadata
- **Fast Queries**: Optimized for analytical queries with proper indexing
- **DataFrame Integration**: Seamless export to pandas DataFrames
- **scikit-learn Ready**: Prepared data structures for machine learning
- **Sample Data Generation**: Tools to generate realistic sample data
- **Analytics Functions**: Built-in analytical queries and visualizations

## Schema

The database schema includes the following fields for job postings:

| Field | Type | Description |
|-------|------|-------------|
| job_id | VARCHAR | Unique identifier for each job posting (Primary Key) |
| title | VARCHAR | Job title |
| company | VARCHAR | Company name |
| location | VARCHAR | Job location |
| salary | VARCHAR | Salary information (may include ranges or currency symbols) |
| description | TEXT | Full job description |
| ai_impact | FLOAT | Score indicating AI's impact on this role (0-1) |
| date_posted | DATE | Date when the job was posted |
| source | VARCHAR | Source of the job posting (e.g., website, job board) |
| created_at | TIMESTAMP | Record creation timestamp |
| updated_at | TIMESTAMP | Record update timestamp |

## Files

- **init_duckdb.py**: Core database initialization and utility functions
- **jobs_analytics_example.py**: Example analytics and visualization with scikit-learn
- **uk_jobs.duckdb**: The DuckDB database file (created when scripts are run)
- **job_analytics_results.png**: Visualization output from the analytics example

## Usage

### Basic Database Operations

```python
from db.init_duckdb import JobsDatabase

# Initialize the database
db = JobsDatabase()
db.initialize_schema()

# Insert a job posting
job_data = {
    "job_id": "JOB-2023-001",
    "title": "Data Scientist",
    "company": "AI Solutions Ltd",
    "location": "London, UK",
    "salary": "£60,000 - £80,000",
    "description": "We are looking for a Data Scientist to join our team...",
    "ai_impact": 0.85,
    "date_posted": "2023-05-15",
    "source": "company_website"
}
db.insert_job(job_data)

# Query jobs
results = db.search_jobs(title="Data Scientist", min_ai_impact=0.7)
for job in results:
    print(f"{job['title']} at {job['company']} - AI Impact: {job['ai_impact']}")

# Export to DataFrame for analysis
df = db.to_dataframe()
print(f"Exported {len(df)} jobs to DataFrame")

# Don't forget to close the connection
db.close()
```

### Analytics and Machine Learning

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from db.init_duckdb import JobsDatabase

# Connect to database
db = JobsDatabase()

# Get data as DataFrame
df = db.to_dataframe()

# Preprocess for machine learning
# Extract text features from job titles
tfidf = TfidfVectorizer(stop_words='english')
title_features = tfidf.fit_transform(df['title'])

# Convert to DataFrame
feature_df = pd.DataFrame(title_features.toarray(), 
                         columns=tfidf.get_feature_names_out())

# Add numerical features
feature_df['ai_impact'] = df['ai_impact']

# Perform clustering
kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(feature_df)

# Analyze clusters
df['cluster'] = clusters
cluster_stats = df.groupby('cluster').agg({
    'ai_impact': ['mean', 'std', 'count'],
    'title': lambda x: pd.Series.mode(x)[0]
})
print(cluster_stats)

db.close()
```

## Advanced Usage

For more advanced usage examples, see the `jobs_analytics_example.py` file, which demonstrates:

1. Generating realistic sample data
2. Creating feature vectors from job data
3. Performing cluster analysis
4. Visualizing AI impact trends
5. Analyzing the distribution of AI impact across different job categories

## Requirements

- Python 3.6+
- DuckDB
- pandas
- numpy
- scikit-learn (for analytics examples)
- matplotlib (for visualizations)

## Installation

```bash
pip install duckdb pandas numpy scikit-learn matplotlib