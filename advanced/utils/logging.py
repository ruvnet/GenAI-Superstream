"""
Logging configuration for the advanced DuckDB implementation.

This module provides logging utilities and configuration for the application.
"""

import os
import logging
import time
from functools import wraps
from typing import Callable, Any

from advanced.config import LOG_CONFIG

# Create logs directory if it doesn't exist
log_dir = os.path.dirname(LOG_CONFIG.get("log_file"))
os.makedirs(log_dir, exist_ok=True)


def setup_logging(name: str = None, level: str = None) -> logging.Logger:
    """
    Set up a logger with the specified name and level.
    
    Args:
        name: Logger name (uses root logger if None)
        level: Logging level (uses level from config if None)
        
    Returns:
        Configured logger
    """
    log_level = getattr(logging, level or LOG_CONFIG.get("level", "INFO"))
    log_format = LOG_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger = logging.getLogger(name)
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Add file handler
    file_handler = logging.FileHandler(LOG_CONFIG.get("log_file"))
    file_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(file_handler)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logger.addHandler(console_handler)
    
    return logger


def timed_function(func: Callable) -> Callable:
    """
    Decorator to time function execution for performance monitoring.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = logging.getLogger(func.__module__)
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.debug(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to execute")
        return result
    
    return wrapper


class LoggedOperation:
    """
    Context manager for logging operations with timing information.
    
    This class provides a way to log the start and end of operations,
    along with timing information and any exceptions that occur.
    """
    
    def __init__(self, operation_name: str, logger: logging.Logger = None):
        """
        Initialize the context manager.
        
        Args:
            operation_name: Name of the operation being performed
            logger: Logger to use (creates new logger if None)
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
    
    def __enter__(self) -> 'LoggedOperation':
        """Enter the context manager and log the start of the operation."""
        self.start_time = time.time()
        self.logger.info(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context manager and log the end of the operation.
        
        Logs timing information and any exceptions that occurred.
        """
        end_time = time.time()
        duration = end_time - self.start_time
        
        if exc_type is not None:
            self.logger.error(f"Operation {self.operation_name} failed after {duration:.2f} seconds: {exc_val}")
            return False
        
        self.logger.info(f"Operation {self.operation_name} completed in {duration:.2f} seconds")
        return True


# Create a log formatter that includes thread ID for concurrent operations
class ThreadAwareFormatter(logging.Formatter):
    """
    Log formatter that includes thread ID and process ID.
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with thread and process information."""
        import threading
        
        record.threadName = threading.current_thread().name
        record.threadID = threading.get_ident()
        
        return super().format(record)


# Setup default root logger
setup_logging()