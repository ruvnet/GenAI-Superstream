#!/usr/bin/env python3
"""
Advanced DuckDB initialization and utility script for UK AI jobs data analytics.

This enhanced module provides comprehensive functionality for creating, managing, and
analyzing a DuckDB database focused on AI's impact on technical roles in the UK job market.
It includes advanced querying capabilities, ML integration, and PerplexityAI data gathering.
"""

import os
import json
import datetime
import logging
import functools
import time
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import pathlib

import duckdb
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Configure logging with advanced settings
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jobs_db.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AIImpactLevel(Enum):
    """Enumeration of AI impact levels for standardization."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    TRANSFORMATIVE = "transformative"


@dataclass
class JobMetrics:
    """Data class for storing job metrics used in analytics."""
    ai_impact_score: float
    salary_range_min: Optional[float]
    salary_range_max: Optional[float]
    salary_median: Optional[float]
    remote_percentage: float
    required_skills_count: int
    job_posting_recency: int  # Days since posting


def timed(func: Callable) -> Callable:
    """Decorator to time function execution for performance monitoring."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper
