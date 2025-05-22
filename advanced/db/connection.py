"""
Database connection management for the advanced DuckDB implementation.

This module provides a database connection manager that handles connecting to DuckDB,
configuring optimizations, and managing the connection lifecycle.
"""

import os
import functools
import logging
from typing import Optional, Any, Callable

import duckdb

from advanced.config import DB_CONFIG, LOG_CONFIG

# Set up logging
logger = logging.getLogger(__name__)


def timed(func: Callable) -> Callable:
    """Decorator to time function execution for performance monitoring."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    return wrapper


class ConnectionManager:
    """
    Manages DuckDB database connections with optimized settings.
    
    This class handles connection lifecycle, configuration, and provides
    a context manager interface for safe usage.
    """
    
    _instance = None  # Singleton instance
    
    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern for connection manager."""
        if cls._instance is None:
            cls._instance = super(ConnectionManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize the connection manager.
        
        Args:
            db_path: Path to the DuckDB database file. If None, uses the path from config.
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return
            
        self.db_path = db_path or DB_CONFIG.get("db_path")
        self.conn = None
        self._query_cache = {}
        self.cache_enabled = DB_CONFIG.get("cache_enabled", True)
        self.cache_size = DB_CONFIG.get("cache_size", 100)
        self._initialized = True
        
        # Create directory for database if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        logger.info(f"ConnectionManager initialized with database path: {self.db_path}")
    
    @timed
    def connect(self) -> duckdb.DuckDBPyConnection:
        """
        Establish a connection to the DuckDB database with optimized settings.
        
        Returns:
            DuckDB connection object
        """
        if self.conn is not None:
            return self.conn
            
        try:
            # Connect to the database
            self.conn = duckdb.connect(self.db_path)
            
            # Configure DuckDB for better performance with analytics workloads
            memory_limit = DB_CONFIG.get("memory_limit", "4GB")
            threads = DB_CONFIG.get("threads", 4)
            
            self.conn.execute(f"SET memory_limit='{memory_limit}'")
            self.conn.execute(f"SET threads={threads}")
            
            logger.info(f"Connected to database at {self.db_path} with optimized settings")
            return self.conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._query_cache.clear()
        logger.debug("Query cache cleared")
    
    def __enter__(self) -> duckdb.DuckDBPyConnection:
        """Context manager entry point."""
        return self.connect()
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        self.close()
    
    @timed
    def execute(self, query: str, parameters: Optional[tuple] = None) -> Any:
        """
        Execute a SQL query with optional parameters.
        
        Args:
            query: SQL query string
            parameters: Query parameters as a tuple
            
        Returns:
            Query result
        """
        conn = self.connect()
        try:
            if parameters:
                return conn.execute(query, parameters)
            return conn.execute(query)
        except Exception as e:
            logger.error(f"Query execution failed: {query[:100]}... - {e}")
            raise
    
    @timed
    def execute_with_cache(self, query: str, parameters: Optional[tuple] = None) -> Any:
        """
        Execute a SQL query with caching support.
        
        Args:
            query: SQL query string
            parameters: Query parameters as a tuple
            
        Returns:
            Query result
        """
        if not self.cache_enabled:
            return self.execute(query, parameters)
            
        # Generate cache key
        cache_key = f"{query}_{str(parameters)}"
        
        # Check if result is in cache
        if cache_key in self._query_cache:
            logger.debug(f"Cache hit for query: {query[:50]}...")
            return self._query_cache[cache_key]
        
        # Execute query
        result = self.execute(query, parameters)
        
        # Cache result
        if len(self._query_cache) >= self.cache_size:
            # Remove oldest entry if cache is full
            self._query_cache.pop(next(iter(self._query_cache)))
        
        self._query_cache[cache_key] = result
        return result
    
    @timed
    def fetch_all(self, query: str, parameters: Optional[tuple] = None) -> list:
        """
        Execute a query and fetch all results.
        
        Args:
            query: SQL query string
            parameters: Query parameters as a tuple
            
        Returns:
            List of query results
        """
        result = self.execute(query, parameters)
        return result.fetchall()
    
    @timed
    def fetch_df(self, query: str, parameters: Optional[tuple] = None) -> 'pd.DataFrame':
        """
        Execute a query and fetch results as a pandas DataFrame.
        
        Args:
            query: SQL query string
            parameters: Query parameters as a tuple
            
        Returns:
            pandas DataFrame with query results
        """
        result = self.execute(query, parameters)
        return result.fetch_df()


# Global connection manager instance
connection_manager = ConnectionManager()