"""
GenAI-Superstream package.

This package provides a machine learning model exposed through a Model Context Protocol (MCP) server,
allowing AI agents to interact with the model through a standardized interface.
"""

__version__ = '0.1.0'

# Import key components for easier access
from src.data import DataLoader, DataPreprocessor
from src.model import ModelTrainer, Predictor, ModelFactory
from src.server import create_mcp_server, MCPServer
from src.client import create_mcp_client, MCPClient
from src.utils import Logger, ConfigManager

# Make these classes available directly from the package
__all__ = [
    'DataLoader', 
    'DataPreprocessor',
    'ModelTrainer', 
    'Predictor', 
    'ModelFactory',
    'MCPServer', 
    'create_mcp_server',
    'MCPClient', 
    'create_mcp_client',
    'Logger', 
    'ConfigManager'
]