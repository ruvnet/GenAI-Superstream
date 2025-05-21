"""
MCP Connector for the GenAI-Superstream project.

This module provides a connector to the API server for MCP integration,
ensuring that predictions can be made reliably without the 'NoneType' error.
"""

import os
import requests
import json
from typing import List, Dict, Any, Optional, Union

from src.utils import Logger

# Initialize logger
logger = Logger().get_logger()

class MCPConnector:
    """
    Connector for MCP integration that communicates with the standalone API server.
    
    This class provides a reliable way to make predictions through the API server,
    avoiding the 'NoneType' object is not callable error in MCP tool execution.
    """
    
    def __init__(self, api_url="http://localhost:8000"):
        """
        Initialize the MCP connector.
        
        Args:
            api_url (str): Base URL for the API server
        """
        self.api_url = api_url
        logger.info(f"Initialized MCP connector with API URL: {api_url}")
    
    def predict_species(self, features: List[float]) -> Dict[str, float]:
        """
        Predict Iris species using the API server.
        
        This method serves as a drop-in replacement for the direct predict_species function,
        ensuring that MCP tool calls work reliably.
        
        Args:
            features (list): List of 4 features [sepal_length, sepal_width, petal_length, petal_width]
            
        Returns:
            dict: Dictionary mapping class names to probabilities
        """
        try:
            # Validate input
            if not isinstance(features, list) or len(features) != 4:
                raise ValueError("Features must be a list of 4 numeric values.")
            
            # Make API request
            endpoint = f"{self.api_url}/api/predict_species"
            payload = {"features": features}
            
            logger.info(f"Making prediction request to {endpoint} with features: {features}")
            
            response = requests.post(
                endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            
            # Check for successful response
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Received prediction: {result}")
                return result
            else:
                error_msg = f"API request failed with status {response.status_code}: {response.text}"
                logger.error(error_msg)
                return {"error": error_msg}
                
        except Exception as e:
            logger.error(f"Error in prediction request: {e}")
            return {"error": str(e)}
    
    def check_health(self) -> bool:
        """
        Check if the API server is available and healthy.
        
        Returns:
            bool: True if the server is healthy, False otherwise
        """
        try:
            endpoint = f"{self.api_url}/health"
            response = requests.get(endpoint, timeout=2)
            
            if response.status_code == 200:
                logger.info("API server health check passed")
                return True
            else:
                logger.warning(f"API server health check failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"API server health check error: {e}")
            return False

# Global connector instance for easy access
_connector = None

def get_connector(api_url="http://localhost:8000") -> MCPConnector:
    """
    Get or create the global connector instance.
    
    Args:
        api_url (str): Base URL for the API server
        
    Returns:
        MCPConnector: The connector instance
    """
    global _connector
    if _connector is None:
        _connector = MCPConnector(api_url)
    return _connector

def predict_species_proxy(features: List[float]) -> Dict[str, float]:
    """
    Proxy function for predict_species to use in MCP tool.
    
    This can be registered as an MCP tool and will forward requests to the API server.
    
    Args:
        features (list): List of 4 features [sepal_length, sepal_width, petal_length, petal_width]
        
    Returns:
        dict: Dictionary mapping class names to probabilities
    """
    connector = get_connector()
    return connector.predict_species(features)

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test MCP connector')
    parser.add_argument('--api-url', type=str, default="http://localhost:8000", 
                      help='Base URL for the API server')
    
    args = parser.parse_args()
    
    # Create connector
    connector = MCPConnector(args.api_url)
    
    # Check server health
    if connector.check_health():
        print("API server is healthy")
        
        # Make a test prediction
        result = connector.predict_species([5.1, 3.5, 1.4, 0.2])
        print(f"Prediction result: {result}")
    else:
        print("API server is not available")