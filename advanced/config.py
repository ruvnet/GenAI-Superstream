"""
Configuration settings for the advanced DuckDB implementation.

This module centralizes all configurable parameters for the application, making it
easier to modify settings without changing code across multiple files.
It also loads environment variables from the .env file if present.
"""

import os
import pathlib
from typing import Dict, Any

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    env_path = os.path.join(pathlib.Path(__file__).parent.absolute(), '.env')
    load_dotenv(env_path)
    print(f"Loaded environment variables from {env_path}")
except ImportError:
    print("python-dotenv not installed, using environment variables as is")
except Exception as e:
    print(f"Failed to load .env file: {e}")

# Base directory for the project
BASE_DIR = pathlib.Path(__file__).parent.absolute()

# Database configuration
DB_CONFIG = {
    "db_path": os.getenv("DB_PATH", os.path.join(BASE_DIR, "duckdb_advance.db")),
    "memory_limit": os.getenv("DB_MEMORY_LIMIT", "4GB"),  # DuckDB memory limit
    "threads": int(os.getenv("DB_THREADS", "4")),         # Number of threads for parallel processing
    "cache_enabled": os.getenv("DB_CACHE_ENABLED", "True").lower() in ("true", "1", "yes"),  # Enable query caching
    "cache_size": int(os.getenv("DB_CACHE_SIZE", "100"))  # Maximum number of queries to cache
}

# Logging configuration
LOG_CONFIG = {
    "log_file": os.getenv("LOG_FILE", os.path.join(BASE_DIR, "logs", "jobs_db.log")),
    "level": os.getenv("LOG_LEVEL", "INFO"),        # Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    "format": os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
}

# PerplexityAI configuration
PERPLEXITY_CONFIG = {
    "server_name": os.getenv("PERPLEXITY_MCP_SERVER_NAME", "perplexityai"),
    "mcp_url": os.getenv("PERPLEXITY_MCP_URL", "http://localhost:3001"),
    "system_prompt": (
        "You are a technical data analyst specializing in the UK job market. "
        "Provide detailed, structured information about AI jobs in the UK focusing on "
        "technical roles. Format data in a way that can be easily parsed for database insertion."
    ),
    "max_tokens": int(os.getenv("PERPLEXITY_MAX_TOKENS", "1000")),
    "temperature": float(os.getenv("PERPLEXITY_TEMPERATURE", "0.1")),  # Lower temperature for more deterministic responses
    "return_citations": os.getenv("PERPLEXITY_RETURN_CITATIONS", "True").lower() in ("true", "1", "yes")
}

# Analytics configuration
ANALYTICS_CONFIG = {
    "export_dir": os.path.join(BASE_DIR, "exports"),
    "visualization_dir": os.path.join(BASE_DIR, "visualizations"),
    "default_cluster_count": 4,
    "default_feature_count": 100,
    "pca_components": 2
}

# Default query parameters
DEFAULT_QUERY_PARAMS = {
    "limit": 100,
    "offset": 0
}

# Sample AI skills for extraction
AI_SKILLS = [
    "Natural Language Processing", "NLP", "Computer Vision", "CV", 
    "Reinforcement Learning", "RL", "Neural Networks", "Transformers",
    "BERT", "GPT", "LLM", "Large Language Models", "Prompt Engineering",
    "RAG", "Retrieval Augmented Generation", "Vector Databases",
    "LangChain", "LlamaIndex", "Hugging Face", "Model Fine-tuning",
    "Model Deployment", "Model Monitoring", "MLOps"
]

# Sample technical skills for extraction
TECH_SKILLS = [
    "Python", "Java", "JavaScript", "TypeScript", "C#", "C++", "Go", "Rust",
    "SQL", "NoSQL", "MongoDB", "PostgreSQL", "MySQL", "DynamoDB",
    "AWS", "Azure", "GCP", "Docker", "Kubernetes", "Terraform",
    "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "scikit-learn",
    "Data Science", "Data Engineering", "ETL", "Spark", "Hadoop",
    "React", "Angular", "Vue", "Node.js", "Django", "Flask", "Spring",
    "DevOps", "CI/CD", "Git", "GitHub", "GitLab"
]

# Create log directory if it doesn't exist
os.makedirs(os.path.join(BASE_DIR, "logs"), exist_ok=True)
os.makedirs(ANALYTICS_CONFIG["export_dir"], exist_ok=True)
os.makedirs(ANALYTICS_CONFIG["visualization_dir"], exist_ok=True)