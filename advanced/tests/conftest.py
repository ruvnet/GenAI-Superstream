"""
Pytest configuration and fixtures for the advanced module tests.

This module provides shared fixtures and configuration for all test modules.
"""

import pytest
import datetime
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from advanced.models.data_classes import (
    JobPosting, Skill, Salary, AIImpactLevel, ContractType, 
    SeniorityLevel, SkillCategory, PerplexityResponse, JobMetrics
)
from advanced.perplexity.client import PerplexityClient
from advanced.perplexity.data_processor import DataProcessor


@pytest.fixture
def sample_job_posting():
    """Create a sample JobPosting for testing."""
    return JobPosting(
        job_id="TEST-001",
        title="Machine Learning Engineer",
        company="TechCorp UK",
        location="London, UK",
        description="We are looking for a talented ML engineer...",
        date_posted=datetime.date(2025, 1, 15),
        source="test_source",
        ai_impact=0.8,
        salary_text="£60,000 - £80,000",
        salary=Salary(min_value=60000.0, max_value=80000.0, currency="GBP"),
        remote_work=True,
        remote_percentage=50,
        contract_type=ContractType.FULL_TIME,
        seniority_level=SeniorityLevel.MID_LEVEL,
        skills=[
            Skill(name="Python", category=SkillCategory.TECHNICAL, is_required=True, experience_years=3),
            Skill(name="TensorFlow", category=SkillCategory.AI, is_required=True, experience_years=2)
        ]
    )


@pytest.fixture
def sample_perplexity_response():
    """Create a sample PerplexityResponse for testing."""
    return PerplexityResponse(
        query_text="Test query about AI jobs",
        response_id="resp-123",
        content="""
        | Job Title | Company | Location | Salary |
        |-----------|---------|----------|--------|
        | Data Scientist | TechCorp | London | £50k-70k |
        | ML Engineer | AIStart | Manchester | £60k-80k |
        """,
        citations=["https://example.com/job1", "https://example.com/job2"],
        data_retrieval_date=datetime.datetime(2025, 5, 22, 18, 0, 0)
    )


@pytest.fixture
def sample_mcp_response():
    """Create a sample MCP response data for testing."""
    return {
        "data": {
            "response": {
                "id": "resp-456",
                "choices": [
                    {
                        "message": {
                            "content": """
                            Here are some AI jobs in the UK:
                            
                            Job Title: Senior Data Scientist
                            Company: DataCorp Ltd
                            Location: London, UK
                            Salary: £70,000 - £90,000
                            Description: Leading data science projects...
                            """
                        }
                    }
                ],
                "citations": ["https://jobs.example.com/1", "https://jobs.example.com/2"]
            }
        }
    }


@pytest.fixture
def mock_environment_variables():
    """Mock environment variables for testing."""
    env_vars = {
        "PERPLEXITY_MCP_SERVER_NAME": "test_perplexity",
        "PERPLEXITY_MCP_URL": "http://test-localhost:3001",
        "PERPLEXITY_MAX_TOKENS": "500",
        "PERPLEXITY_TEMPERATURE": "0.2",
        "DB_PATH": "/tmp/test.db",
        "DB_MEMORY_LIMIT": "2GB",
        "DB_THREADS": "2",
        "LOG_LEVEL": "DEBUG"
    }
    with patch.dict(os.environ, env_vars):
        yield env_vars


@pytest.fixture
def mock_dotenv():
    """Mock the dotenv loading functionality."""
    with patch('advanced.config.load_dotenv') as mock_load:
        mock_load.return_value = None
        yield mock_load


@pytest.fixture
def perplexity_client():
    """Create a PerplexityClient instance for testing."""
    return PerplexityClient(server_name="test_server")


@pytest.fixture
def data_processor():
    """Create a DataProcessor instance for testing."""
    return DataProcessor()


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing."""
    mock_conn = MagicMock()
    mock_conn.execute = MagicMock()
    mock_conn.fetch_all = MagicMock()
    mock_conn.__enter__ = MagicMock(return_value=mock_conn)
    mock_conn.__exit__ = MagicMock(return_value=None)
    return mock_conn


@pytest.fixture
def sample_table_data():
    """Sample table data for data extraction testing."""
    return """
    | Job Title | Company | Location | Salary | AI Impact |
    |-----------|---------|----------|--------|-----------|
    | Data Scientist | TechCorp | London | £50k-70k | High |
    | ML Engineer | AIStart | Manchester | £60k-80k | Very High |
    | Backend Developer | CodeCorp | Bristol | £45k-60k | Medium |
    """


@pytest.fixture
def sample_json_data():
    """Sample JSON data for data extraction testing."""
    return """
    [
        {
            "title": "AI Research Scientist",
            "company": "Research Labs",
            "location": "Cambridge, UK",
            "salary": "£80,000 - £100,000",
            "ai_impact": "transformative"
        },
        {
            "title": "Data Engineer",
            "company": "Data Solutions",
            "location": "Edinburgh, UK", 
            "salary": "£55,000 - £75,000",
            "ai_impact": "medium"
        }
    ]
    """


@pytest.fixture
def sample_key_value_data():
    """Sample key-value data for data extraction testing."""
    return """
    Title: Senior Python Developer
    Company: DevCorp Ltd
    Location: Liverpool, UK
    Salary: £65,000 - £85,000
    Requirements: 5+ years Python experience
    
    Title: Frontend Developer
    Company: WebTech Solutions
    Location: Remote, UK
    Salary: £40,000 - £55,000
    Requirements: React, TypeScript experience
    """


@pytest.fixture
def temporary_config_file():
    """Create a temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("""
PERPLEXITY_MCP_SERVER_NAME=test_server
PERPLEXITY_MAX_TOKENS=1000
DB_PATH=/tmp/test.db
LOG_LEVEL=INFO
        """)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    if os.path.exists(temp_file):
        os.unlink(temp_file)


@pytest.fixture
def mock_argparse_namespace():
    """Create a mock argparse namespace for CLI testing."""
    from argparse import Namespace
    return Namespace(
        init=False,
        gather=True,
        stats=False,
        query=None,
        role="Data Scientist",
        location="London",
        timeframe="last month",
        batch_size=5,
        dry_run=False
    )


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    import logging
    # Remove all handlers from root logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Reset level
    root_logger.setLevel(logging.WARNING)
    yield
    
    # Cleanup after test
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)


@pytest.fixture
def mock_pandas_dataframe():
    """Mock pandas DataFrame for testing."""
    import pandas as pd
    data = {
        'job_id': ['TEST-001', 'TEST-002'],
        'title': ['Data Scientist', 'ML Engineer'],
        'company': ['TechCorp', 'AIStart'],
        'location': ['London, UK', 'Manchester, UK'],
        'ai_impact': [0.8, 0.9]
    }
    return pd.DataFrame(data)